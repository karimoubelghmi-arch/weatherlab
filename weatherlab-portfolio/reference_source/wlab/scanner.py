"""Scan multiville + couche trading (paper).

Pour chaque ville : prévisions live (Open-Meteo J-1) → chaîne → proba YES de chaque bucket
(la proba du modèle), puis comparaison au marché Polymarket (edge) et dimensionnement Kelly.

Kelly (vérifié) : pour un contrat acheté au prix p qui paie 1, de proba de gain pw,
la fraction optimale est f* = (pw − p)/(1 − p) = edge/(1 − p). On prend le côté (YES/NO)
qui a un edge positif, puis on plafonne l'exposition globale.
"""
from __future__ import annotations

from dataclasses import dataclass

import json
import numpy as np
import pandas as pd

from .data import city_frame, models_of
from .engine import run_pipeline, dist_label, cfg_for_city
from .lead import lead_hours, sigma_scale, scale_for as lead_scale_for, fetch_moment
from .live import forecast_source
from .live import fetch_forecasts, fetch_forecasts_with_peaks, fetch_model_tmax


def target_err_prev_for(slug, spec, models, target_date, live_cache, obs, fc):
    """V11 — « erreur d'hier » au moment du scan : obs(J) − moyenne des prévisions(J),
    où J = veille de la cible = AUJOURD'HUI (heure locale ville). obs(J) : socle si déjà
    intégré, sinon max METAR du jour (station officielle). fc(J) : socle si présent,
    sinon fetch previous_day1 (mis en cache). Renvoie (err|None, note)."""
    import pandas as pd, numpy as np
    y_date = pd.Timestamp(target_date) - pd.Timedelta(days=1)
    # --- prévisions d'aujourd'hui ---
    fcv = None
    try:
        rows = fc[(fc["slug"] == slug) & (fc["date"] == y_date) & (fc["model"].isin(models))]
        if len(rows) >= max(2, len(models) - 1):
            fcv = float(rows["forecast_c"].mean())
    except Exception:
        pass
    if fcv is None:
        entry = (live_cache or {}).get(slug) or {}
        cached = entry.get("fc_today")
        if cached and cached.get("date") == str(y_date.date()):
            vals = [v for v in cached["fc"].values() if v is not None]
            fcv = float(np.mean(vals)) if vals else None
        else:
            try:
                got = fetch_forecasts(models, spec["lat"], spec["lon"], spec["tz"], y_date.date())
                vals = [v for v in got.values() if v is not None]
                fcv = float(np.mean(vals)) if vals else None
                if live_cache is not None:
                    live_cache.setdefault(slug, {}).update(
                        {"fc_today": {"date": str(y_date.date()), "fc": got}})
            except Exception:
                fcv = None
    if fcv is None:
        return None, "prévisions du jour indisponibles"
    # --- observation d'aujourd'hui ---
    ov = None
    try:
        o = obs[(obs["slug"] == slug) & (obs["date"] == y_date)]
        if len(o):
            ov = float(o["tmax_c"].iloc[0])
            return ov - fcv, "socle"
    except Exception:
        pass
    try:
        from .live_obs import fetch_metar
        from zoneinfo import ZoneInfo
        from datetime import datetime
        icao = spec.get("icao")
        now_loc = datetime.now(ZoneInfo(spec["tz"]))
        # garde-fou : la Tmax du jour n'est fiable que si le pic est passé (soirée locale).
        # Avant ~17 h locale, le « max du jour » METAR serait une température de nuit/matinée
        # comparée à une prévision d'après-midi → erreur absurde. Repli neutre 0.
        if now_loc.date() != y_date.date() or now_loc.hour < 17:
            return None, f"pic du jour pas encore passé ({now_loc:%H h} locale) → scanner en soirée"
        if icao:
            m = fetch_metar(icao, hours=30)
            if m is not None and len(m):
                loc = m["time_utc"].dt.tz_convert(ZoneInfo(spec["tz"]))
                today = loc.dt.date == y_date.date()
                if today.any() and int(loc[today].dt.hour.max()) >= 15:
                    ov = float(m.loc[today, "temp_c"].max())
                    return ov - fcv, "METAR (max du jour, pic passé)"
    except Exception:
        pass
    return None, "observation du jour indisponible"
from .polymarket import compare
from .liquidity import fetch_books, max_stake_for_avg_price, exec_stats


def attach_books(candidates: pd.DataFrame, workers: int = 8, progress=None) -> pd.DataFrame:
    """Ajoute la colonne `book_asks` (carnet côté vente du token acheté) à chaque candidat.

    Un seul appel réseau par token, en parallèle. Les carnets sont ensuite réutilisés
    par build_candidates à chaque changement de stratégie, SANS nouvel appel réseau.
    """
    if candidates.empty or "tokenId" not in candidates.columns:
        return candidates
    books = fetch_books(candidates["tokenId"].astype(str), workers=workers, progress=progress)
    out = candidates.copy()
    out["book_asks"] = out["tokenId"].astype(str).map(books)
    return out


@dataclass
class TradingConfig:
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25      # fraction de Kelly (0.25 = quart de Kelly)
    min_edge: float = 0.03            # edge minimal. NB : plus de shrinkage séparé — si tu veux
                                      # compenser un winner's curse mesuré au backtest, ajoute-le
                                      # directement ici (ex. 8 pts + 1 pt de curse → 9 pts).
    min_trade: float = 1.0            # mise minimale retenue après plafonnement
    side: str = "both"                # "both" | "YES" | "NO" — côté(s) autorisé(s)
    per_order_max_usd: float = 5.0    # 🔒 KILL-SWITCH / PARI : mise max par ordre ($).
                                      # Une seule limite par ordre (l'ancien « cap/pari » a fusionné
                                      # dedans) ; appliquée au dimensionnement ET par l'exécuteur.
    daily_max_usd: float = 25.0       # 🔒 KILL-SWITCH 24 h : dépense max par jour ($).
                                      # Une seule limite globale (l'ancienne « exposition max » a
                                      # fusionné dedans) ; appliquée au dimensionnement ET par
                                      # l'exécuteur (fenêtre 24 h glissantes).
    blend_w: float = 1.0              # mélange log-odds : poids du MODÈLE (1.0 = pas de mélange).
                                      # p̃ = σ(w·logit(p_modèle) + (1−w)·logit(p_marché)).
                                      # Punit surtout les désaccords extrêmes (queues) ;
                                      # un désaccord modéré est peu raboté.


def blend_logodds(p_model, p_market, w: float):
    """Mélange log-odds de la proba modèle vers la proba marché (vectorisé).

    w = poids du modèle ∈ [0,1]. w=1 → p_model inchangée ; w=0 → on croit le marché.
    Contrairement au mélange linéaire (≡ Kelly fractionné), le mélange en logit
    tire d'autant plus fort vers le marché que le désaccord est extrême — c'est le
    comportement voulu pour les paris de queue (modèle 20 % vs marché 1 %).
    Probas bornées à [0.5 %, 99.5 %] pour éviter les logits infinis.
    """
    w = float(min(1.0, max(0.0, w)))
    if w >= 1.0:
        return np.asarray(p_model, dtype=float)
    p = np.clip(np.asarray(p_model, dtype=float), 0.005, 0.995)
    q = np.clip(np.asarray(p_market, dtype=float), 0.005, 0.995)
    lo = w * np.log(p / (1 - p)) + (1 - w) * np.log(q / (1 - q))
    return 1.0 / (1.0 + np.exp(-lo))


def kelly_fraction(edge_side: float, price_side: float) -> float:
    p = min(0.99, max(0.01, float(price_side)))
    f = float(edge_side) / (1.0 - p)
    return max(0.0, min(1.0, f))


# ----------------------------------------------------------------------------- #
#  SCAN
# ----------------------------------------------------------------------------- #
def scan_cities(obs, fc, meta, slugs, target_dates, pipeline_cfg, fetch_market=True, progress=None,
                live_cache=None):
    """Renvoie (model_buckets, candidates, infos, errors).

    target_dates : dict {slug -> date locale de la ville} (timezone-aware, géré par l'appelant).
    live_cache   : cache de prévisions (st.session_state["live_cache"], rempli par le bouton
                   « Récupérer les dernières prévisions »). Si une ville y figure pour la BONNE
                   date, ses prévisions sont réutilisées telles quelles (aucun appel Open-Meteo) ;
                   sinon on ne télécharge que les modèles manquants et on complète le cache.

    - model_buckets : DataFrame long (ville, slug, température, unité, proba_YES_modèle_%)
      = la proba YES de chaque bucket pour CHAQUE ville (cœur de la demande).
    - candidates    : DataFrame des contrats avec marché (model%, marché%, edge, côté), vide si pas de marché.
    - infos         : par ville, μ/σ/loi/date + source des prévisions (cache ou Open-Meteo).
    - errors        : (ville, raison) pour les villes non traitées.
    """
    mb_rows, cand_rows, infos, errors = [], [], [], []
    for i, slug in enumerate(slugs):
        name = meta[slug]["name"]
        if progress:
            progress(i, len(slugs), name)
        try:
            date = target_dates[slug]                          # date LOCALE de la ville
            models = models_of(meta, slug, only_used=True)
            spec = meta[slug]

            # 1) réutiliser le cache si l'entrée correspond à la date cible
            live = {}
            cached = (live_cache or {}).get(slug)
            peak_fc = dict((cached or {}).get("peak_fc", {}))     # V10 : pics prévus
            if cached and str(cached.get("date")) == str(date):
                live = {m: cached.get("fc", {}).get(m) for m in models}
            to_fetch = [m for m in models if live.get(m) is None]
            src = "cache" if not to_fetch else ("Open-Meteo" if len(to_fetch) == len(models)
                                                else "cache + Open-Meteo")

            # 2) ne télécharger QUE ce qui manque, et compléter le cache
            if to_fetch:
                fetched, fetched_peaks = fetch_forecasts_with_peaks(to_fetch, spec["lat"],
                                                                    spec["lon"], spec["tz"], date)
                live.update(fetched)
                peak_fc.update({m: h for m, h in fetched_peaks.items() if h is not None})
                if live_cache is not None:
                    entry = live_cache.get(slug)
                    if not entry or str(entry.get("date")) != str(date):
                        entry = {"date": str(date), "fc": {}}
                        live_cache[slug] = entry
                    entry["fc"].update({m: v for m, v in fetched.items() if v is not None})
                    entry.setdefault("peak_fc", {}).update(
                        {m: h for m, h in fetched_peaks.items() if h is not None})
                    # v9.1 : horodatage de la recuperation (reference du facteur d'echeance)
                    entry["fetched_ts"] = pd.Timestamp.now(tz="UTC").isoformat()

            miss = [m for m in models if live.get(m) is None]
            if miss:
                if len(miss) == len(models):
                    try:
                        fetch_model_tmax(models[0], spec["lat"], spec["lon"], spec["tz"], date)
                        reason = "aucune valeur renvoyée pour cette date"
                    except Exception as exc:
                        reason = f"réseau/SSL : {str(exc)[:90]}"
                    errors.append((name, f"prévisions live indisponibles ({reason})"))
                else:
                    errors.append((name, f"prévisions partielles, modèles manquants : {', '.join(miss)}"))
                continue
            df = city_frame(obs, fc, slug, models)
            # ---- V10 : cohérence d'échéance (méthode BEST uniquement) ----
            lead_info = None
            _cfg_city = cfg_for_city(pipeline_cfg, slug)
            if getattr(pipeline_cfg, "lead_coherent", False) and pipeline_cfg.reg.method == "BEST":
                from dataclasses import replace as _replace
                from .engine import model_scores
                from .best_model import lead_coherent_frame
                _col = "MAE_°C" if pipeline_cfg.reg.best_metric == "MAE" else "RMSE_°C"
                _best = str(model_scores(df, models).sort_values(_col).iloc[0]["modèle"])
                df, lead_info = lead_coherent_frame(
                    df, slug, spec, _best, date, peak_fc.get(_best),
                    tol=float(getattr(pipeline_cfg, "lead_tol", 1.0)))
                if lead_info.get("applique"):
                    _cfg_city = _replace(_cfg_city, reg=_replace(_cfg_city.reg, forced_model=_best))
            tf = {m: float(live[m]) for m in models}          # déjà en °C
            # ---- V11 : « erreur d'hier » au moment du scan ----
            err_prev, err_note = (None, "désactivée")
            if getattr(pipeline_cfg.reg, "use_err_prev", False) and pipeline_cfg.reg.method != "BEST":
                err_prev, err_note = target_err_prev_for(slug, spec, models, date, live_cache, obs, fc)
            doy = int(pd.Timestamp(date).day_of_year)
            # facteur d'echeance (v9) : calcule ET applique via un point d'entree unique.
            # v9.1 : la reference est le moment de RECUPERATION des previsions de cette
            # ville (cache), pas l'heure courante.
            _moment = fetch_moment((live_cache or {}).get(slug))
            ld = lead_scale_for(slug, spec["tz"], pd.Timestamp(date).date(),
                                enabled=bool(getattr(pipeline_cfg, "use_lead_scale", True)),
                                now=_moment)
            lh, scale = ld["lead_h"], ld["scale"]
            # V13 : ancre J-2 si calibrée pour la ville (voir wlab/anchor.py).
            # Passée à run_pipeline UNIQUEMENT quand le facteur d'échéance s'applique
            # (source « courante ») — sinon comportement v9 inchangé.
            from wlab.anchor import get as anchor_get
            _anchor = anchor_get(slug) if ld["applique"] else None
            ctx = run_pipeline(df, models, spec["settlement_unit"], _cfg_city,
                               target_forecasts=tf, target_doy=doy, sigma_scale=scale,
                               lead_h=(lh if ld["applique"] else None), anchor=_anchor,
                               target_err_prev=err_prev)
            infos.append({"ville": name, "slug": slug, "date_cible (locale)": str(date),
                          "🕐_échéance": ((f"{lead_info['lead_cible_h']:.0f}h "
                                          f"(pic {lead_info['pic_utilisé']}h, ±{lead_info['tol_h']:.0f}h, "
                                          f"{lead_info['n_strate']}/{lead_info['n_total']} j)")
                                         if lead_info and lead_info.get("applique")
                                         else ("poolé" if lead_info is None else
                                               f"poolé ({lead_info.get('pic_source', '?')})")),
                          "μ_°C": round(ctx["mu"], 2),
                          "err_hier_°C": (round(err_prev, 2) if err_prev is not None else f"— ({err_note})"),
                          "a": round(float(ctx["a"]), 3), "b": round(float(ctx["b"]), 3),
                          "σ_base_°C": round(ctx["sigma_base"], 3),
                          "échéance_h": (round(lh, 1) if lh is not None else None),
                          "σ×": round(ctx["sigma_scale"], 3),
                          "σ_°C": round(ctx["sigma"], 3),
                          "ancre_J2": ("—" if not ctx.get("anchor") else
                                       f"R={ctx['anchor']['ratio']:.2f} w={ctx['anchor']['w']:.2f}"
                                       + ("" if ctx["anchor"]["mu_blend"] else " (μ₁)")),
                          "spread": round(ctx["spread"], 2),
                          "loi": ctx["law"]["best"],
                          # V20 : photo COMPLÈTE pour l'audit — paramètres ajustés de la
                          # loi, dispersion des z, et prévisions par modèle (JSON).
                          "loi_params": dist_label(ctx["dist"]),
                          "std_z": round(float(ctx["dist"].std()), 3),
                          "prévisions_modèles": json.dumps({m: round(float(tf[m]), 2) for m in models},
                                                           ensure_ascii=False),
                          "unité": spec["settlement_unit"],
                          "critère_loi": ctx["law"].get("criterion", "BIC"),
                          "prévisions": f"{src} ({forecast_source()})",
                          "réf_échéance": ld.get("reference", "heure courante"),
                          "σ×_appliqué": ("oui" if ld["applique"] else f"non — {ld['raison']}")})

            # proba YES modèle par bucket entier
            u = spec["settlement_unit"]
            for _, r in ctx["buckets"].iterrows():
                if r["proba"] >= 0.005:
                    mb_rows.append({"ville": name, "slug": slug, "température": int(r[f"température_{u}"]),
                                    "unité": u, "proba_YES_modèle_%": round(r["proba"] * 100, 1)})

            # comparaison marché
            if fetch_market:
                cmp = compare(slug, date, ctx["mu"], ctx["sigma"], ctx["dist"])
                if "statut" in cmp.columns:
                    errors.append((name, f"marché : {cmp.iloc[0]['statut']}"))
                else:
                    for _, r in cmp.iterrows():
                        cand_rows.append({"ville": name, "slug": slug, "date": str(date), "contrat": r["contrat"],
                                          "proba_modèle_%": r["proba_modèle_%"], "prix_marché_%": r["prix_marché_%"],
                                          "edge_YES_%": r["edge_YES_%"], "côté": r["côté"],
                                          "prix_côté_%": r.get("prix_côté_%"), "tokenId": r.get("tokenId", ""),
                                          "conditionId": r.get("conditionId", ""), "market_id": r.get("market_id", ""),
                                          "negRisk": r.get("negRisk", False)})
        except Exception as exc:
            errors.append((meta[slug]["name"], str(exc)[:120]))
    return (pd.DataFrame(mb_rows), pd.DataFrame(cand_rows), pd.DataFrame(infos), errors)


# ----------------------------------------------------------------------------- #
#  CANDIDATS DE TRADING (Kelly + plafond d'exposition)
# ----------------------------------------------------------------------------- #
def build_candidates(candidates: pd.DataFrame, cfg: TradingConfig) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    df = candidates.copy()
    # côté retenu + prix/edge du côté
    df["edge_côté_%"] = df["edge_YES_%"].abs()

    # Prix du côté tradé. On force une Series avec le même index pour rester
    # compatible avec toutes les versions de pandas : fillna n'accepte pas
    # toujours un ndarray retourné par np.where.
    prix_calc = pd.Series(
        np.where(df["côté"] == "YES", df["prix_marché_%"], 100 - df["prix_marché_%"]),
        index=df.index,
    )
    if "prix_côté_%" not in df.columns or df["prix_côté_%"].isna().all():
        df["prix_côté_%"] = prix_calc
    else:
        df["prix_côté_%"] = df["prix_côté_%"].fillna(prix_calc)

    df["proba_côté_%"] = np.where(df["côté"] == "YES", df["proba_modèle_%"], 100 - df["proba_modèle_%"])
    side = str(getattr(cfg, "side", "both")).upper()
    if side in ("YES", "NO"):
        df = df[df["côté"] == side].copy()
        if df.empty:
            return df
    # ---- MÉLANGE LOG-ODDS (optionnel) ----
    # La proba OPÉRATIONNELLE devient p̃ (glissée vers le marché) : c'est elle qui
    # alimente edge, filtre, Kelly ET la couche liquidité plus bas. La proba brute
    # du modèle reste visible dans proba_modèle_côté_%.
    w = float(getattr(cfg, "blend_w", 1.0))
    df["proba_modèle_côté_%"] = df["proba_côté_%"]
    if w < 1.0:
        df["proba_côté_%"] = (blend_logodds(df["proba_côté_%"] / 100.0,
                                            df["prix_côté_%"] / 100.0, w) * 100).round(2)
    # edge du côté recalculé DIRECTEMENT sur (proba_côté, prix_côté) — plus honnête
    # que |edge_YES| pour un pari NO quand prix_NO ≠ 100 − prix_YES (spread).
    df["edge_côté_%"] = df["proba_côté_%"] - df["prix_côté_%"]
    df = df[df["edge_côté_%"] >= cfg.min_edge * 100].copy()
    if df.empty:
        return df
    f = df.apply(lambda r: kelly_fraction(r["edge_côté_%"] / 100, r["prix_côté_%"] / 100), axis=1)
    df["kelly_%"] = (f * 100).round(1)
    # 🔒 kill-switch / pari : la mise Kelly est plafonnée directement en $
    raw = (f * cfg.kelly_fraction * cfg.bankroll).clip(upper=float(cfg.per_order_max_usd))
    # 🔒 kill-switch 24 h : le total du scan est rescalé sous le plafond quotidien
    tot = raw.sum()
    day_cap = float(cfg.daily_max_usd)
    scale = 1.0 if tot <= day_cap or tot <= 0 else day_cap / tot
    df["mise_kelly_$"] = (raw * scale).round(2)   # ce que la stratégie voulait miser

    # ---- couche LIQUIDITÉ : critère d'edge EFFECTIF (moyen) ----
    # 1) mise_kelly_$ = mise voulue par la stratégie (Kelly × fraction, cap/pari,
    #    rescale d'exposition globale). C'est « combien j'aurais misé ».
    # 2) liq_utile_$ = mise maximale dont le PRIX MOYEN d'exécution garde un edge
    #    effectif ≥ edge_min (on peut mordre au-dessus de la limite tant que la
    #    moyenne tient).
    # 3) mise_$ = min(mise_kelly, liq_utile).
    # Statuts distincts pour ne PAS confondre les causes :
    #   🚫 bloqué (liquidité)  : liq_utile < mise min → même la mise minimale aurait
    #                            un edge effectif < edge_min. C'est le carnet qui bloque.
    #   💤 sous mise min       : la mise Kelly elle-même (après cap/rescale d'exposition)
    #                            est < mise min. Rien à voir avec le carnet.
    has_books = "book_asks" in df.columns
    liq_l, pmoy_l, eeff_l, statut_l, mise_l = [], [], [], [], []
    for _, r in df.iterrows():
        mise = float(r["mise_kelly_$"])
        asks = r.get("book_asks") if has_books else None
        if not isinstance(asks, list) or not asks:
            liq_l.append(None); pmoy_l.append(None); eeff_l.append(None)
            if mise < cfg.min_trade:
                statut_l.append("💤 sous mise min"); mise_l.append(0.0)
            else:
                statut_l.append("❔ inconnue"); mise_l.append(mise)
            continue
        # la liquidité est jugée sur l'edge opérationnel : prix moyen ≤ proba − edge_min.
        p_model = float(r["proba_côté_%"]) / 100.0
        liq = max_stake_for_avg_price(asks, p_model - cfg.min_edge)
        liq_l.append(round(liq, 2))
        if liq < cfg.min_trade:
            # le CARNET bloque : même la mise minimale aurait un edge effectif < edge_min.
            # On montre ce qu'aurait donné la mise Kelly, pour comprendre pourquoi.
            _, avg, _ = exec_stats(asks, mise if mise >= cfg.min_trade else cfg.min_trade)
            pmoy_l.append(round(avg * 100, 1) if avg else None)
            eeff_l.append(round(r["proba_côté_%"] - avg * 100, 1) if avg else None)
            statut_l.append("🚫 bloqué (liquidité)"); mise_l.append(0.0)
            continue
        if mise < cfg.min_trade:
            _, avg, _ = exec_stats(asks, max(mise, 0.01))
            pmoy_l.append(round(avg * 100, 1) if avg else None)
            eeff_l.append(round(r["proba_côté_%"] - avg * 100, 1) if avg else None)
            statut_l.append("💤 sous mise min"); mise_l.append(0.0)
            continue
        mise_ok = min(mise, liq)
        _, avg, _ = exec_stats(asks, mise_ok)
        pmoy_l.append(round(avg * 100, 1) if avg else None)
        eeff_l.append(round(r["proba_côté_%"] - avg * 100, 1) if avg else None)
        statut_l.append("✂️ écrêtée" if mise_ok < mise - 0.005 else "✅ ok")
        mise_l.append(round(mise_ok, 2))
    df["liq_utile_$"] = liq_l
    df["prix_moy_exéc_%"] = pmoy_l
    df["edge_effectif_%"] = eeff_l
    df["liquidité"] = statut_l
    df["mise_$"] = mise_l

    df = df.sort_values("edge_côté_%", ascending=False).reset_index(drop=True)
    cols = ["ville", "slug", "date", "contrat", "côté", "proba_modèle_%", "proba_modèle_côté_%", "proba_côté_%",
            "prix_marché_%", "prix_côté_%",
            "edge_côté_%", "edge_effectif_%", "prix_moy_exéc_%", "liq_utile_$", "liquidité",
            "kelly_%", "mise_kelly_$", "mise_$", "tokenId", "conditionId", "market_id", "negRisk"]
    return df[[c for c in cols if c in df.columns]]
