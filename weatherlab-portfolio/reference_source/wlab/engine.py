"""Moteur WeatherLab 2 — toute la chaîne, recalculée À PARTIR DES DONNÉES.

📡 Prévisions → 🧮 Régression μ → 📏 Spread → 📉 σ → 🔬 Loi de E → 🌡️ Distribution T → 🎯 Probas

Chaque étape est une fonction pure + une dataclass de config (donc tout est modifiable
depuis l'UI). Rien n'est lu depuis d'anciens params figés : on refitte sur l'historique.

Convention : tout en °C. T = μ + σ·Z, Z ~ loi de E. P(T≤x) = loi.cdf((x−μ)/σ).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------- #
#  CONFIGS (tous les leviers modifiables)
# ----------------------------------------------------------------------------- #
@dataclass
class RegConfig:
    method: str = "BEST"           # "BEST" (V9 : un seul modèle, le meilleur) | "OLS" | "Ridge"
    ridge_lambda: float = 1.0
    use_season: bool = True        # ajoute sin/cos(jour de l'année) — ignoré si method="BEST"
    # --- V9 : paramètres du mode "BEST" (un seul modèle) --------------------------------
    best_metric: str = "MAE"       # "MAE" | "RMSE" : critère de choix du meilleur modèle
    bias_correct: bool = False     # μ = T_best − biais moyen (intercept) au lieu de μ = T_best
    forced_model: str | None = None  # V10 : force CE modèle (utilisé par la cohérence d'échéance)
    use_err_prev: bool = True        # V11 : feature « erreur d'hier » dans μ (OLS/Ridge uniquement).
                                     # Gain validé hors-échantillon : ΔCRPS −0.019 (t=−5.8) sur 49 villes.


@dataclass
class SigmaConfig:
    spread_mode: str = "max-min"   # "max-min" | "écart-type" | "MAD"
    a: float | None = None         # None => ajusté sur |résidu| ; sinon valeur forcée
    b: float | None = None
    k: float = 1.0                 # facteur de conversion (cancellé par le fit de la loi ; cf. README)
    sigma_min: float = 0.5


@dataclass
class LawConfig:
    family: str = "auto"           # "auto" => choix automatique ; sinon famille forcée
    candidates: tuple = ("Normal", "Student-t", "Laplace", "Skew-normal", "Skew-t")
    criterion: str = "BIC"         # "BIC" | "AIC" (dans l'échantillon) | "logscore" | "bucketll" (hors-échantillon)


@dataclass
class PipelineConfig:
    reg: RegConfig = field(default_factory=RegConfig)
    sigma: SigmaConfig = field(default_factory=SigmaConfig)
    law: LawConfig = field(default_factory=LawConfig)
    test_fraction: float = 0.30    # part finale de l'historique réservée à la validation
    use_lead_scale: bool = True    # applique le facteur d'échéance σ× (wlab/lead.py) à la
                                   # distribution prédictive. Réf. 24 h = variable de
                                   # calibration → ×1.0 ; <24 h → σ rétréci ; >24 h → élargi.
    # V9 : définition du spread PAR VILLE. {slug: "max-min"|"écart-type"|"MAD"} ; une ville
    # absente du dict suit sigma.spread_mode (réglage global). Résolution via cfg_for_city.
    spread_overrides: dict = field(default_factory=dict)
    # V10 : cohérence d'échéance (expérimental, méthode BEST uniquement) — la calibration
    # σ + loi n'utilise que les jours historiques dont l'échéance (pic − run previous_day1)
    # est proche de celle attendue DEMAIN (pic prévu par la série live). ±lead_tol heures,
    # tolérance élargie automatiquement si la strate est trop petite.
    lead_coherent: bool = False
    lead_tol: float = 0.5   # ±0.5 h = échéance exacte à l'heure près (élargie auto si <80 j)


def cfg_for_city(cfg: "PipelineConfig", slug: str) -> "PipelineConfig":
    """Copie de cfg avec la définition du spread RÉSOLUE pour la ville `slug`
    (override par ville si présent, sinon réglage global). À appeler dans toute
    boucle multiville avant run_pipeline / validate."""
    mode = (cfg.spread_overrides or {}).get(slug)
    if not mode or mode == cfg.sigma.spread_mode:
        return cfg
    from dataclasses import replace
    return replace(cfg, sigma=replace(cfg.sigma, spread_mode=mode))


# ----------------------------------------------------------------------------- #
#  📅  Saison
# ----------------------------------------------------------------------------- #
def season_terms(doy):
    ang = 2.0 * math.pi * np.asarray(doy, dtype=float) / 365.25
    return np.sin(ang), np.cos(ang)


# ----------------------------------------------------------------------------- #
#  🧮  Régression μ
# ----------------------------------------------------------------------------- #
def _design(model_matrix: np.ndarray, doy, use_season: bool, err_prev=None):
    n = model_matrix.shape[0]
    cols = [model_matrix]
    names = []
    if err_prev is not None:                       # V11 : feature « erreur d'hier »
        cols.append(np.asarray(err_prev, float).reshape(-1, 1))
        names += ["err_prev"]
    if use_season:
        s, c = season_terms(doy)
        cols.append(np.column_stack([s, c]))
        names += ["sin", "cos"]
    cols.append(np.ones((n, 1)))
    A = np.column_stack(cols)
    return A, names


def model_scores(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """V9 — MAE / RMSE / biais de CHAQUE modèle individuel contre la vérité (tmax_c).

    Calculé sur les lignes COMPLÈTES de `df` (mêmes jours pour tous les modèles →
    comparaison équitable). biais = moyenne(prévision − vérité) : >0 = modèle trop chaud."""
    y = df["tmax_c"].to_numpy(float)
    rows = []
    for m in models:
        e = df[m].to_numpy(float) - y
        rows.append({"modèle": m,
                     "MAE_°C": float(np.mean(np.abs(e))),
                     "RMSE_°C": float(np.sqrt(np.mean(e ** 2))),
                     "biais_°C": float(np.mean(e)),
                     "n_jours": int(len(e))})
    return pd.DataFrame(rows).sort_values("MAE_°C").reset_index(drop=True)


def fit_regression(df: pd.DataFrame, models: list[str], cfg: RegConfig) -> dict:
    """Ajuste μ, selon cfg.method :

    • "BEST" (V9, défaut) : PAS de combinaison. Un seul modèle est retenu — celui qui a
      la plus petite MAE (ou RMSE, cfg.best_metric) sur `df` — et μ = sa prévision brute
      (ou débiaisée d'une constante si cfg.bias_correct). Techniquement le vecteur β est
      un one-hot (1 sur le modèle retenu, 0 ailleurs), si bien que TOUTE la chaîne aval
      (predict, validation, scan, audit) fonctionne à l'identique.
    • "OLS" / "Ridge" (hérité) : μ = Σ β·modèle (+ saison) + intercept en °C."""
    M = df[models].to_numpy(float)
    y = df["tmax_c"].to_numpy(float)

    if cfg.method == "BEST":
        scores = model_scores(df, models)
        col = "MAE_°C" if getattr(cfg, "best_metric", "MAE") == "MAE" else "RMSE_°C"
        _forced = getattr(cfg, "forced_model", None)
        best = _forced if _forced in models else str(scores.sort_values(col).iloc[0]["modèle"])
        i_best = models.index(best)
        # design SANS saison (le mode BEST n'utilise jamais sin/cos) : [modèles | 1]
        A, extra = _design(M, df["doy"].to_numpy(), False)
        beta = np.zeros(A.shape[1])
        beta[i_best] = 1.0
        if getattr(cfg, "bias_correct", False):
            beta[-1] = float(np.mean(y - M[:, i_best]))   # intercept = −biais moyen
        yhat = A @ beta
        resid = y - yhat
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        names = list(models) + extra + ["intercept"]
        return {
            "coefs": dict(zip(names, beta.tolist())),
            "models": list(models),
            "use_season": False,
            "use_err_prev": False,
            "beta_vector": beta,
            "design_names": names,
            "resid": resid,
            "fitted": yhat,
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "mae": float(np.mean(np.abs(resid))),
            "n": len(y),
            "method": "BEST",
            "best_model": best,
            "best_metric": getattr(cfg, "best_metric", "MAE"),
            "bias_correct": bool(getattr(cfg, "bias_correct", False)),
            "model_scores": scores,
        }

    _err = (df["err_prev"].to_numpy(float)
            if (getattr(cfg, "use_err_prev", False) and "err_prev" in df.columns) else None)
    A, extra = _design(M, df["doy"].to_numpy(), cfg.use_season, err_prev=_err)

    if cfg.method == "Ridge":
        # Ridge STANDARDISÉ : on centre/réduit les variables pénalisées (modèles + saison),
        # on centre y, on ajuste, puis on retransforme vers l'échelle d'origine (°C).
        # Sans ça, λ agirait sur l'échelle arbitraire des features et n'aurait quasi aucun effet.
        Xp = A[:, :-1]
        mu_x = Xp.mean(0)
        sd_x = Xp.std(0, ddof=0)
        sd_x[sd_x == 0] = 1.0
        Xs = (Xp - mu_x) / sd_x
        yc = y - y.mean()
        p = Xs.shape[1]
        b_std = np.linalg.solve(Xs.T @ Xs + cfg.ridge_lambda * np.eye(p), Xs.T @ yc)
        b_feat = b_std / sd_x
        intercept = y.mean() - mu_x @ b_feat
        beta = np.concatenate([b_feat, [intercept]])
    else:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)

    yhat = A @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    names = list(models) + extra + ["intercept"]
    coefs = dict(zip(names, beta.tolist()))
    return {
        "coefs": coefs,
        "models": list(models),
        "use_season": cfg.use_season,
        "use_err_prev": _err is not None,
        "beta_vector": beta,
        "design_names": names,
        "resid": resid,
        "fitted": yhat,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "n": len(y),
    }


def predict_mu(reg: dict, model_vals: dict, doy: int, err_prev: float | None = None) -> float:
    c = reg["coefs"]
    mu = sum(c[m] * float(model_vals[m]) for m in reg["models"])
    if reg.get("use_err_prev"):
        mu += c["err_prev"] * float(err_prev if err_prev is not None else 0.0)
    if reg["use_season"]:
        s, cc = season_terms([doy])
        mu += c["sin"] * float(s[0]) + c["cos"] * float(cc[0])
    mu += c["intercept"]
    return float(mu)


# ----------------------------------------------------------------------------- #
#  📏  Spread
# ----------------------------------------------------------------------------- #
def spread_row(values: np.ndarray, mode: str) -> float:
    v = np.asarray(values, float)
    if mode == "écart-type":
        return float(v.std(ddof=1)) if v.size > 1 else 0.0
    if mode == "MAD":
        return float(np.median(np.abs(v - np.median(v))))
    return float(v.max() - v.min())            # max-min par défaut


def spread_series(df: pd.DataFrame, models: list[str], mode: str) -> np.ndarray:
    M = df[models].to_numpy(float)
    if mode == "écart-type":
        return M.std(axis=1, ddof=1)
    if mode == "MAD":
        med = np.median(M, axis=1, keepdims=True)
        return np.median(np.abs(M - med), axis=1)
    return M.max(axis=1) - M.min(axis=1)


# ----------------------------------------------------------------------------- #
#  📉  σ = max(σ_min, (a + b·spread)·k)
# ----------------------------------------------------------------------------- #
def fit_sigma(spread: np.ndarray, abs_resid: np.ndarray, cfg: SigmaConfig) -> tuple[float, float]:
    """Régresse |résidu| sur le spread (OLS) → a, b. Forcés si cfg.a/b fournis."""
    if cfg.a is not None and cfg.b is not None:
        return float(cfg.a), float(cfg.b)
    A = np.column_stack([np.ones_like(spread), spread])
    (a, b), *_ = np.linalg.lstsq(A, abs_resid, rcond=None)
    return float(a), float(b)


def sigma_of(spread, a: float, b: float, cfg: SigmaConfig):
    raw = (a + b * np.asarray(spread, float)) * cfg.k
    return np.maximum(cfg.sigma_min, raw)


# ----------------------------------------------------------------------------- #
#  🔬  Loi de E (sur le résidu standardisé z = résidu / σ)
# ----------------------------------------------------------------------------- #
def _fit_family(z: np.ndarray, fam: str):
    z = np.asarray(z, float)
    if fam == "Normal":
        loc, sc = stats.norm.fit(z);                       return stats.norm(loc, sc), 2
    if fam == "Student-t":
        df, loc, sc = stats.t.fit(z, floc=0) if False else stats.t.fit(z)
        return stats.t(df, loc, sc), 3
    if fam == "Laplace":
        loc, sc = stats.laplace.fit(z);                    return stats.laplace(loc, sc), 2
    if fam == "Skew-normal":
        a, loc, sc = stats.skewnorm.fit(z);                return stats.skewnorm(a, loc, sc), 3
    if fam == "Skew-t":
        if not hasattr(stats, "jf_skew_t"):
            raise ValueError("Skew-t indisponible (scipy<1.12)")
        a, b, loc, sc = stats.jf_skew_t.fit(z);            return stats.jf_skew_t(a, b, loc, sc), 4
    raise ValueError(f"loi inconnue : {fam}")


def select_law(z: np.ndarray, cfg: LawConfig) -> dict:
    """Ajuste chaque famille candidate, classe par BIC. Renvoie le tableau + la retenue."""
    z = np.asarray(z, float)
    n = z.size
    rows = []
    fitted = {}
    for fam in cfg.candidates:
        try:
            dist, kpar = _fit_family(z, fam)
            ll = float(np.sum(dist.logpdf(z)))
            aic = 2 * kpar - 2 * ll
            bic = kpar * math.log(n) - 2 * ll
            rows.append({"loi": fam, "params": kpar, "loglik": ll, "AIC": aic, "BIC": bic})
            fitted[fam] = dist
        except Exception as exc:
            rows.append({"loi": fam, "params": np.nan, "loglik": np.nan, "AIC": np.nan, "BIC": np.nan, "erreur": str(exc)})
    tab = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    if cfg.family != "auto" and cfg.family in fitted:
        best = cfg.family
    else:
        best = tab.iloc[0]["loi"]
    return {"table": tab, "best": best, "dist": fitted[best], "fitted": fitted}


def dist_label(dist) -> str:
    name = type(dist.dist).__name__
    args = ", ".join(f"{x:.3f}" for x in (list(dist.args) + [dist.kwds.get(k) for k in dist.kwds]))
    return f"{name}({args})"


# ----------------------------------------------------------------------------- #
#  🌡️  Distribution T + 🎯 Probabilités de buckets
# ----------------------------------------------------------------------------- #
def cdf_T(x, mu, sigma, dist):
    return dist.cdf((np.asarray(x, float) - mu) / sigma)


def pdf_T(x, mu, sigma, dist):
    return dist.pdf((np.asarray(x, float) - mu) / sigma) / sigma


def c2f(c):
    return c * 1.8 + 32.0


def f2c(f):
    return (f - 32.0) / 1.8


def bucket_table(mu: float, sigma: float, dist, unit: str, radius_sigma: float = 6.0) -> pd.DataFrame:
    """Probabilité de chaque température ENTIÈRE dans l'unité de règlement.
    μ, σ sont en °C ; si l'unité est °F on raisonne sur la grille entière en °F,
    en convertissant chaque borne en °C avant d'intégrer (cohérent avec Polymarket)."""
    half = 0.5
    if unit == "F":
        center = c2f(mu)
        sig_f = sigma * 1.8
        lo = int(math.floor(center - radius_sigma * sig_f))
        hi = int(math.ceil(center + radius_sigma * sig_f))
        to_c = f2c
    else:
        lo = int(math.floor(mu - radius_sigma * sigma))
        hi = int(math.ceil(mu + radius_sigma * sigma))
        to_c = lambda x: x
    rows = []
    for k in range(lo, hi + 1):
        lo_c = to_c(k - half)
        hi_c = to_c(k + half)
        p = float(cdf_T(hi_c, mu, sigma, dist) - cdf_T(lo_c, mu, sigma, dist))
        rows.append({f"température_{unit}": k, "intervalle_°C": f"[{lo_c:.2f}, {hi_c:.2f})", "proba": max(0.0, p)})
    df = pd.DataFrame(rows)
    tot = df["proba"].sum()
    df["proba_%"] = (df["proba"] * 100)
    if tot > 0:
        df["proba_normalisée_%"] = df["proba"] / tot * 100
    return df


# ----------------------------------------------------------------------------- #
#  🔬  Choix de la loi selon un critère (BIC/AIC dans l'échantillon, ou score
#      propre hors-échantillon : log-score de densité, log-loss sur les buckets)
# ----------------------------------------------------------------------------- #
def _oos_law_scores(df, models, cfg, unit, fams):
    """Pour chaque famille : ajuste la loi sur le TRAIN, score sur le TEST (fin d'historique).
    Renvoie {famille: {logscore_oos, bucketLL_oos}}. {} si pas assez de données."""
    n = len(df)
    cut = int(n * (1 - cfg.test_fraction))
    tr, te = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    if len(te) < 20 or len(tr) < 50:
        return {}
    reg = fit_regression(tr, models, cfg.reg)
    sp_tr = spread_series(tr, models, cfg.sigma.spread_mode)
    a, b = fit_sigma(sp_tr, np.abs(reg["resid"]), cfg.sigma)
    z_tr = reg["resid"] / sigma_of(sp_tr, a, b, cfg.sigma)
    Mte = te[models].to_numpy(float)
    Ate, _ = _design(Mte, te["doy"].to_numpy(), reg["use_season"],
                     err_prev=(te["err_prev"].to_numpy(float)
                               if reg.get("use_err_prev") and "err_prev" in te.columns else None))
    mu = Ate @ reg["beta_vector"]
    sig = sigma_of(spread_series(te, models, cfg.sigma.spread_mode), a, b, cfg.sigma)
    y = te["tmax_c"].to_numpy(float)
    # buckets entiers dans l'unité de règlement (°C, ou °F converti)
    if unit == "F":
        yb, mub, sigb = y * 9 / 5 + 32, mu * 9 / 5 + 32, sig * 9 / 5
    else:
        yb, mub, sigb = y, mu, sig
    kstar = np.round(yb).astype(int)
    out = {}
    for f in fams:
        try:
            d, _ = _fit_family(z_tr, f)
            ls = float(-np.mean(np.log(np.clip(pdf_T(y, mu, sig, d), 1e-12, None))))
            p = d.cdf((kstar + 0.5 - mub) / sigb) - d.cdf((kstar - 0.5 - mub) / sigb)
            bll = float(-np.mean(np.log(np.clip(p, 1e-12, None))))
            out[f] = {"logscore_oos": ls, "bucketLL_oos": bll}
        except Exception:
            out[f] = {"logscore_oos": np.inf, "bucketLL_oos": np.inf}
    return out


_CRIT_COL = {"BIC": "BIC", "AIC": "AIC", "logscore": "logscore_oos", "bucketll": "bucketLL_oos"}


def _score_laws_split(tr, te, models, cfg, unit, fams):
    """Ajuste la chaîne sur `tr`, score chaque famille de loi sur `te`.
    Renvoie {famille: (logscore, bucketLL)} — plus bas = mieux ; inf si échec du fit."""
    reg = fit_regression(tr, models, cfg.reg)
    sp_tr = spread_series(tr, models, cfg.sigma.spread_mode)
    a, b = fit_sigma(sp_tr, np.abs(reg["resid"]), cfg.sigma)
    z_tr = reg["resid"] / sigma_of(sp_tr, a, b, cfg.sigma)
    Mte = te[models].to_numpy(float)
    Ate, _ = _design(Mte, te["doy"].to_numpy(), reg["use_season"],
                     err_prev=(te["err_prev"].to_numpy(float)
                               if reg.get("use_err_prev") and "err_prev" in te.columns else None))
    mu = Ate @ reg["beta_vector"]
    sig = sigma_of(spread_series(te, models, cfg.sigma.spread_mode), a, b, cfg.sigma)
    y = te["tmax_c"].to_numpy(float)
    if unit == "F":
        yb, mub, sigb = y * 9 / 5 + 32, mu * 9 / 5 + 32, sig * 9 / 5
    else:
        yb, mub, sigb = y, mu, sig
    kstar = np.round(yb).astype(int)
    out = {}
    for f in fams:
        try:
            d, _ = _fit_family(z_tr, f)
            ls = float(-np.mean(np.log(np.clip(pdf_T(y, mu, sig, d), 1e-12, None))))
            p = d.cdf((kstar + 0.5 - mub) / sigb) - d.cdf((kstar - 0.5 - mub) / sigb)
            bll = float(-np.mean(np.log(np.clip(p, 1e-12, None))))
            out[f] = (ls, bll)
        except Exception:
            out[f] = (float("inf"), float("inf"))
    return out


def law_robustness(df, models, cfg, unit=None, n_splits: int = 10, test_frac: float = 0.15,
                   progress=None) -> dict:
    """Test de ROBUSTESSE du choix de loi : rolling-origin sur n_splits découpages.

    Pour chaque point de coupe c (échelonné entre 50 % de l'historique et la fin moins
    une fenêtre de test) : train = tout avant c, test = bloc contigu suivant
    (~test_frac de l'historique). Chaque famille est ajustée sur le train et notée sur
    le test (log-score + score buckets Polymarket). Une loi n'est « adoptée » que si
    elle bat la Normale de façon RÉPÉTÉE à travers les splits — pas sur un seul
    découpage chanceux. (Même philosophie que l'« étape 5 » de l'ancien bot.)
    """
    n = len(df)
    len_te = max(40, int(round(n * test_frac)))
    first_cut = max(120, int(n * 0.5))
    last_cut = n - len_te
    if last_cut <= first_cut:
        return {"ok": False, "msg": "Historique trop court pour un test multi-splits fiable."}
    cuts = np.unique(np.linspace(first_cut, last_cut, int(n_splits)).astype(int))
    fams = list(cfg.law.candidates)
    rows = []
    for i, c in enumerate(cuts):
        tr = df.iloc[:c].reset_index(drop=True)
        te = df.iloc[c:c + len_te].reset_index(drop=True)
        sc = _score_laws_split(tr, te, models, cfg, unit, fams)
        fin = str(pd.Timestamp(df.iloc[c - 1]["date"]).date()) if "date" in df.columns else str(c)
        for f, (ls, bll) in sc.items():
            rows.append({"split": i + 1, "fin_train": fin, "n_train": c, "loi": f,
                         "logscore": ls, "bucketLL": bll})
        if progress:
            progress(i + 1, len(cuts))
    longd = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

    summ = (longd.groupby("loi")
                 .agg(bucketLL_moy=("bucketLL", "mean"), bucketLL_std=("bucketLL", "std"),
                      logscore_moy=("logscore", "mean"), logscore_std=("logscore", "std"))
                 .reset_index())
    wins_b = longd.loc[longd.groupby("split")["bucketLL"].idxmin(), "loi"].value_counts()
    wins_l = longd.loc[longd.groupby("split")["logscore"].idxmin(), "loi"].value_counts()
    summ["victoires_buckets"] = summ["loi"].map(wins_b).fillna(0).astype(int)
    summ["victoires_logscore"] = summ["loi"].map(wins_l).fillna(0).astype(int)
    piv = longd.pivot_table(index="split", columns="loi", values="bucketLL")
    if "Normal" in piv.columns:
        deltas = {f: (float((piv["Normal"] - piv[f]).mean()),
                      float(((piv["Normal"] - piv[f]) > 0).mean())) for f in piv.columns}
        summ["gain_vs_Normal"] = summ["loi"].map(lambda f: round(deltas[f][0], 4))
        summ["bat_Normal_%"] = summ["loi"].map(lambda f: round(deltas[f][1] * 100))
    summ = summ.sort_values("bucketLL_moy").reset_index(drop=True)
    for cc in ("bucketLL_moy", "bucketLL_std", "logscore_moy", "logscore_std"):
        summ[cc] = summ[cc].round(4)

    best = summ.iloc[0]["loi"]
    if best == "Normal":
        verdict = "✅ Garder Normal : c'est déjà la meilleure loi en moyenne sur les splits."
    else:
        pct = float(summ.loc[summ["loi"] == best, "bat_Normal_%"].iloc[0]) if "bat_Normal_%" in summ.columns else 0.0
        gain = float(summ.loc[summ["loi"] == best, "gain_vs_Normal"].iloc[0]) if "gain_vs_Normal" in summ.columns else 0.0
        if pct >= 70 and gain > 0:
            verdict = (f"✅ Adopter {best} : bat la Normale sur {pct:.0f} % des splits "
                       f"(gain moyen {gain:+.4f} en score buckets — robuste).")
        else:
            verdict = (f"⚠️ Garder Normal : le gain de {best} n'est PAS robuste "
                       f"(ne bat la Normale que sur {pct:.0f} % des splits).")
    return {"ok": True, "détail": longd, "résumé": summ, "verdict": verdict,
            "n_splits": int(len(cuts)), "len_test": int(len_te)}


def select_law_ex(df, models, cfg, unit=None, z=None) -> dict:
    """Comme select_law mais le classement suit cfg.law.criterion.

    BIC/AIC : dans l'échantillon (rapide). logscore/bucketll : hors-échantillon (split
    70/30, plus honnête pour la prévision et le pricing des buckets Polymarket). La loi
    retenue est ensuite ajustée sur TOUT l'historique. Le tableau renvoyé contient toutes
    les colonnes disponibles pour comparaison."""
    lcfg = cfg.law
    if z is None:
        reg = fit_regression(df, models, cfg.reg)
        sp = spread_series(df, models, cfg.sigma.spread_mode)
        a, b = fit_sigma(sp, np.abs(reg["resid"]), cfg.sigma)
        z = reg["resid"] / sigma_of(sp, a, b, cfg.sigma)
    z = np.asarray(z, float)
    n = z.size
    rows, fitted = [], {}
    for fam in lcfg.candidates:
        try:
            d, k = _fit_family(z, fam)
            ll = float(np.sum(d.logpdf(z)))
            rows.append({"loi": fam, "params": k, "loglik": ll, "AIC": 2 * k - 2 * ll, "BIC": k * math.log(n) - 2 * ll})
            fitted[fam] = d
        except Exception as exc:
            rows.append({"loi": fam, "params": np.nan, "loglik": np.nan, "AIC": np.nan, "BIC": np.nan, "erreur": str(exc)})
    tab = pd.DataFrame(rows)

    crit = getattr(lcfg, "criterion", "BIC")
    if crit in ("logscore", "bucketll"):
        sc = _oos_law_scores(df, models, cfg, unit, list(fitted.keys()))
        if sc:
            tab["logscore_oos"] = tab["loi"].map(lambda f: sc.get(f, {}).get("logscore_oos", np.nan))
            tab["bucketLL_oos"] = tab["loi"].map(lambda f: sc.get(f, {}).get("bucketLL_oos", np.nan))
        else:
            crit = "BIC"  # repli si historique trop court pour un split fiable

    sortcol = _CRIT_COL.get(crit, "BIC")
    if sortcol not in tab.columns:
        sortcol = "BIC"
    tab = tab.sort_values(sortcol, na_position="last").reset_index(drop=True)
    if lcfg.family != "auto" and lcfg.family in fitted:
        best = lcfg.family
    else:
        best = tab.iloc[0]["loi"]
    return {"table": tab, "best": best, "dist": fitted[best], "fitted": fitted,
            "criterion": crit, "sort_col": sortcol}


# ----------------------------------------------------------------------------- #
#  ✅  Validation (honnête : split temporel train/test)
# ----------------------------------------------------------------------------- #
def _predictive_sigma(df, models, reg, scfg, a, b):
    sp = spread_series(df, models, scfg.spread_mode)
    return sigma_of(sp, a, b, scfg)


def validate(df: pd.DataFrame, models: list[str], cfg: PipelineConfig, unit=None) -> dict:
    """Refait toute la chaîne sur le TRAIN puis évalue sur le TEST (fin de l'historique).
    Métriques : couverture des intervalles, log-score, CRPS, PIT (pour l'histogramme).
    Le choix de la loi suit cfg.law.criterion (sélection nichée sur le train, sans fuite)."""
    n = len(df)
    cut = int(n * (1 - cfg.test_fraction))
    tr, te = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    if len(te) < 20 or len(tr) < 50:
        return {"ok": False, "msg": "Pas assez de données pour un split fiable."}

    reg = fit_regression(tr, models, cfg.reg)
    sp_tr = spread_series(tr, models, cfg.sigma.spread_mode)
    a, b = fit_sigma(sp_tr, np.abs(reg["resid"]), cfg.sigma)
    sig_tr = sigma_of(sp_tr, a, b, cfg.sigma)
    z_tr = reg["resid"] / sig_tr
    law = select_law_ex(tr, models, cfg, unit=unit, z=z_tr)
    dist = law["dist"]

    # TEST : μ et σ prédits hors-échantillon
    Mte = te[models].to_numpy(float)
    Ate, _ = _design(Mte, te["doy"].to_numpy(), reg["use_season"],
                     err_prev=(te["err_prev"].to_numpy(float)
                               if reg.get("use_err_prev") and "err_prev" in te.columns else None))
    mu_te = Ate @ reg["beta_vector"]
    sp_te = spread_series(te, models, cfg.sigma.spread_mode)
    sig_te = sigma_of(sp_te, a, b, cfg.sigma)
    y = te["tmax_c"].to_numpy(float)

    # PIT + log-score
    u = cdf_T(y, mu_te, sig_te, dist)
    logscore = float(-np.mean(np.log(np.clip(pdf_T(y, mu_te, sig_te, dist), 1e-12, None))))
    # CRPS numérique (grille commune large)
    lo = float(min(y.min(), mu_te.min()) - 8)
    hi = float(max(y.max(), mu_te.max()) + 8)
    grid = np.linspace(lo, hi, 600)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    crps_list = []
    for yi, mi, si in zip(y, mu_te, sig_te):
        F = dist.cdf((grid - mi) / si)
        H = (grid >= yi).astype(float)
        crps_list.append(_trapz((F - H) ** 2, grid))
    crps = float(np.mean(crps_list))
    cover = {f"{int(q*100)}%": float(np.mean(np.abs(u - 0.5) <= q / 2)) for q in (0.5, 0.8, 0.9)}

    return {
        "ok": True, "n_train": len(tr), "n_test": len(te),
        "a": a, "b": b, "law": law["best"], "law_table": law["table"],
        "logscore": logscore, "crps": crps, "pit": u, "coverage": cover,
        "mu_test": mu_te, "sigma_test": sig_te, "y_test": y, "dates_test": te["date"].values,
    }


# ----------------------------------------------------------------------------- #
#  🔗  Pipeline complet pour UNE ville (fit sur tout l'historique)
# ----------------------------------------------------------------------------- #
def run_pipeline(df: pd.DataFrame, models: list[str], unit: str, cfg: PipelineConfig,
                 target_forecasts: dict | None = None, target_doy: int | None = None,
                 sigma_scale: float = 1.0, lead_h: float | None = None,
                 anchor: dict | None = None, target_err_prev: float | None = None) -> dict:
    """Ajuste la chaîne sur tout l'historique de la ville, puis produit la distribution
    prédictive pour une journée cible (dernière ligne par défaut, ou prévisions manuelles)."""
    reg = fit_regression(df, models, cfg.reg)
    sp = spread_series(df, models, cfg.sigma.spread_mode)
    a, b = fit_sigma(sp, np.abs(reg["resid"]), cfg.sigma)
    sig_hist = sigma_of(sp, a, b, cfg.sigma)
    z = reg["resid"] / sig_hist
    law = select_law_ex(df, models, cfg, unit=unit, z=z)
    dist = law["dist"]

    # journée cible
    if target_forecasts is None:
        last = df.iloc[-1]
        target_forecasts = {m: float(last[m]) for m in models}
        target_doy = int(last["doy"])
        target_date = pd.Timestamp(last["date"]).date()
        if target_err_prev is None and reg.get("use_err_prev") and "err_prev" in df.columns:
            target_err_prev = float(last["err_prev"])
    else:
        target_date = None
    err_prev_defaulted = bool(reg.get("use_err_prev")) and target_err_prev is None
    mu = predict_mu(reg, target_forecasts, target_doy, err_prev=target_err_prev)
    sp_t = spread_row(np.array([target_forecasts[m] for m in models]), cfg.sigma.spread_mode)
    sigma_base = float(sigma_of(sp_t, a, b, cfg.sigma))

    # --- facteur d'echeance (v9) -------------------------------------------------
    # La calibration a ete faite sur la prevision de la VEILLE (lead ~24 h). Si
    # l'information utilisee est plus fraiche (J+1 a 19,6 h par exemple) la vraie
    # dispersion est plus faible : on multiplie sigma par le facteur interpole AVANT
    # de construire la distribution et les buckets. C'est bien sigma_final qui sert a
    # tout le reste de la chaine (densite, probas, edge, Kelly).
    scale = float(sigma_scale)
    if not getattr(cfg, "use_lead_scale", True):
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    # --- V13 : ancre J-2 (voir wlab/anchor.py) --------------------------------
    # Si une ancre calibrée existe et que l'échéance dépasse 24 h, le facteur et μ
    # sont pris sur l'ancre : ratio σ₂(s)/σ₁(s) conditionnel au spread DU JOUR
    # (supprime le double comptage spread×facteur) et β₂ mélangés dans μ
    # (les poids 24 h ne sont pas les bons poids 48 h). Sous 24 h : grille v9.
    anchor_info = None
    if (anchor is not None and lead_h is not None and float(lead_h) > 24.0
            and getattr(cfg, "use_lead_scale", True)):
        try:
            from .anchor import adjust as _anchor_adjust
            anchor_info = _anchor_adjust(anchor, float(lead_h), mu, sp_t, a, b,
                                         cfg.sigma, target_forecasts, int(target_doy),
                                         alpha_tail=anchor.get("alpha_tail"))
            scale = float(anchor_info["scale"])
            mu = float(anchor_info["mu"])
        except Exception:
            anchor_info = None
    sigma = sigma_base
    if scale != 1.0:
        # jamais sous le plancher sigma_min de la config
        sigma = max(float(cfg.sigma.sigma_min), sigma_base * scale)
    buckets = bucket_table(mu, sigma, dist, unit)

    return {
        "reg": reg,
        "err_prev": (float(target_err_prev) if target_err_prev is not None else None),
        "err_prev_defaulted": err_prev_defaulted, "a": a, "b": b, "spread_hist": sp, "sigma_hist": sig_hist,
        "z": z, "law": law, "dist": dist,
        "target_forecasts": target_forecasts, "target_doy": target_doy, "target_date": target_date,
        "mu": mu, "spread": sp_t,
        "sigma": sigma,                 # <- sigma REELLEMENT utilise (apres facteur)
        "sigma_base": sigma_base,       # <- sigma de calibration (echeance 24 h)
        "sigma_scale": scale,           # <- facteur applique (1.0 = aucun)
        "anchor": anchor_info,          # <- V13 : detail de l'ancre J-2 (None = non ancre)
        "unit": unit, "buckets": buckets,
        "models": models, "cfg": cfg,
    }
