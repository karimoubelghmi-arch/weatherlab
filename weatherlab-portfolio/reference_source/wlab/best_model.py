"""V9 — Choix du MEILLEUR modèle + cohérence horaire des runs (previous_day1).

Deux rôles :

1)  Scores individuels des modèles (MAE / RMSE / biais) — le calcul lui-même vit dans
    wlab.engine.model_scores ; ici on ajoute la vue multiville et les utilitaires UI.

2)  Analyse horaire : le socle est construit avec l'API « Previous Runs » d'Open-Meteo,
    variable *_previous_day1. Ce n'est PAS « le run de 17 h la veille » : pour chaque
    heure valide H, previous_day1 sert la valeur prédite ≥ 24 h avant, c'est-à-dire par
    le DERNIER run assimilé au plus tard à H − 24 h. Concrètement les 24 premières
    heures de chaque run sont ignorées ; pour un modèle mis à jour toutes les 6 h, le
    run initialisé à R couvre les heures valides [R+24 h, R+30 h[.

    Donc si le pic de température d'un jour est à 17 h locale, la valeur previous_day1
    à 17 h vient du run initialisé à l'heure UTC = plancher(17 h_locale_en_UTC − 24 h)
    sur la grille de runs du modèle → l'échéance réelle est entre 24 h et
    24 h + cadence. Chaque modèle ayant une cadence différente, l'échéance réelle
    varie de quelques heures d'un modèle à l'autre : ce module la rend explicite.

    L'heure du pic des jours PASSÉS n'est pas dans le socle (qui n'a que la Tmax
    quotidienne) : on la reconstruit depuis les relevés HORAIRES de la STATION
    OFFICIELLE de règlement (METAR — les mêmes relevés que Wunderground affiche —
    via l'archive ASOS de l'Iowa Environmental Mesonet, par code ICAO, en heure
    locale), puis on prend l'argmax de chaque journée. Repli ERA5 (Open-Meteo)
    seulement si la station est indisponible. Cache dans data/peak_hours/.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .data import DATA_DIR, city_frame, models_of
from .engine import model_scores

PEAK_DIR = DATA_DIR / "peak_hours"

# ----------------------------------------------------------------------------- #
#  🕐  Cadence des runs par modèle (grille d'initialisation UTC, telle
#      qu'ingérée par Open-Meteo — source : docs Open-Meteo / model updates).
#      cadence_h = pas entre deux runs ingérés. Certains « seamless » mélangent
#      un modèle global (cadence lente) et un modèle local (cadence rapide) :
#      la note le précise.
# ----------------------------------------------------------------------------- #
MODEL_RUNS: dict[str, dict] = {
    #                              grille d'initialisation UTC explicite (init_hours)
    "ecmwf_ifs025":               {"init_hours": [0, 6, 12, 18], "note": "IFS 00/06/12/18 UTC"},
    "gfs_seamless":               {"init_hours": [0, 6, 12, 18], "note": "GFS 00/06/12/18 UTC (HRRR horaire intégré sur les USA)"},
    "gfs_hrrr":                   {"init_hours": list(range(24)), "note": "HRRR : run toutes les heures (USA)"},
    "icon_global":                {"init_hours": [0, 6, 12, 18], "note": "ICON 00/06/12/18 UTC"},
    "icon_eu":                    {"init_hours": list(range(0, 24, 3)), "note": "ICON-EU : run toutes les 3 h"},
    "gem_global":                 {"init_hours": [0, 12], "note": "GEM global 00/12 UTC — échéance jusqu'à 36 h !"},
    "gem_hrdps_continental":      {"init_hours": [0, 6, 12, 18], "note": "HRDPS 00/06/12/18 UTC (Canada)"},
    "jma_seamless":               {"init_hours": [0, 6, 12, 18], "note": "JMA GSM 00/06/12/18 UTC (MSM 3 h sur le Japon)"},
    "meteofrance_arpege_europe":  {"init_hours": [0, 6, 12, 18], "note": "ARPEGE 00/06/12/18 UTC"},
    "ukmo_seamless":              {"init_hours": [0, 6, 12, 18], "note": "UKMO global 00/06/12/18 UTC (UKV horaire sur le UK)"},
    "knmi_harmonie_arome_europe": {"init_hours": list(range(24)), "note": "KNMI Harmonie : run toutes les heures"},
    "dmi_harmonie_arome_europe":  {"init_hours": list(range(0, 24, 3)), "note": "DMI Harmonie : run toutes les 3 h"},
    "metno_nordic":               {"init_hours": list(range(24)), "note": "MET Norway Nordic : run toutes les heures"},
    "cma_grapes_global":          {"init_hours": [0, 6, 12, 18], "note": "CMA 00/06/12/18 UTC"},
}
_DEFAULT_INIT_HOURS = [0, 6, 12, 18]

# Délai de MISE À DISPOSITION après l'init (calcul + diffusion + ingestion Open-Meteo).
# Docs Open-Meteo : ~4–6 h pour les globaux, ~1–3 h pour les régionaux. Valeurs
# CONSERVATRICES (borne haute) — c'est un délai de sécurité, pas une horloge exacte.
_AVAIL_DELAY_H = {
    "ecmwf_ifs025": 6, "gfs_seamless": 6, "icon_global": 6, "gem_global": 6,
    "jma_seamless": 6, "ukmo_seamless": 6, "cma_grapes_global": 6,
    "icon_eu": 3, "meteofrance_arpege_europe": 4, "gem_hrdps_continental": 3,
    "dmi_harmonie_arome_europe": 3, "knmi_harmonie_arome_europe": 2,
    "metno_nordic": 2, "gfs_hrrr": 2,
}


def _cadence_of(init_hours: list[int]) -> float:
    """Pas MAXIMAL entre deux runs consécutifs (= pire écart d'échéance possible)."""
    hs = sorted(init_hours)
    gaps = [(hs[(i + 1) % len(hs)] - hs[i]) % 24 or 24 for i in range(len(hs))]
    return float(max(gaps)) if hs else 24.0


def run_for_valid_hour(model: str, valid_local: datetime, tz: str) -> dict:
    """Run utilisé par previous_day1 pour l'heure valide `valid_local` (naïve, heure locale).

    previous_day1 ignore les 24 premières heures de chaque run : la valeur vient du
    DERNIER run initialisé au plus tard à (valide_UTC − 24 h), sur la GRILLE EXPLICITE
    de runs du modèle. lead_h ∈ [24, 24 + écart max de la grille[ — ex. gem_global
    (00/12 UTC) peut monter à 36 h ; un pic à 23 h UTC donne le run de 12 h la
    veille → 35 h d'échéance, pendant qu'un modèle horaire est à 24 h pile."""
    ih = MODEL_RUNS.get(model, {}).get("init_hours", _DEFAULT_INIT_HOURS)
    v_utc = valid_local.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)
    thr = v_utc - timedelta(hours=24)
    # dernier init ≤ thr sur la grille (regarde le jour de thr puis les jours d'avant)
    init = None
    for back in range(0, 3):
        day = (thr - timedelta(days=back)).replace(minute=0, second=0, microsecond=0)
        cands = [day.replace(hour=h) for h in sorted(ih) if day.replace(hour=h) <= thr]
        if cands:
            init = max(cands)
            break
    lead = (v_utc - init).total_seconds() / 3600.0
    return {"init_utc": init, "lead_h": lead, "cadence_h": _cadence_of(ih)}


def daily_leads(models: list[str], peaks: pd.DataFrame, tz: str) -> pd.DataFrame:
    """V9 — l'échéance RÉELLE de chaque OBSERVATION du socle : pour chaque jour,
    échéance = heure du pic observée (station) − run utilisé par previous_day1.
    Une colonne par modèle (lead_<modèle>, en heures), une ligne par date."""
    out = {"date": [], "peak_hour": []}
    for m in models:
        out[f"lead_{m}"] = []
    for r in peaks.itertuples():
        d = pd.Timestamp(r.date).to_pydatetime().replace(hour=int(r.peak_hour), minute=0)
        out["date"].append(pd.Timestamp(r.date))
        out["peak_hour"].append(int(r.peak_hour))
        for m in models:
            out[f"lead_{m}"].append(run_for_valid_hour(m, d, tz)["lead_h"])
    return pd.DataFrame(out)


def lead_history(models: list[str], peaks: pd.DataFrame, tz: str,
                 df: pd.DataFrame | None = None) -> pd.DataFrame:
    """V9 — l'échéance RÉELLE historique de chaque modèle, jour par jour, en utilisant
    l'heure du pic OBSERVÉE (station) de chaque journée. C'est la quantification exacte
    de l'incohérence horaire de l'ancien système multi-modèles : un même jour, la
    « prévision de la veille » de chaque modèle n'avait pas la même fraîcheur.

    Si `df` (city_frame : date, tmax_c, colonnes modèles) est fourni, ajoute le test
    empirique « l'échéance du jour explique-t-elle l'erreur ? » : MAE séparée sur les
    jours d'échéance COURTE (≤ médiane du modèle) vs LONGUE (> médiane)."""
    dl = daily_leads(models, peaks, tz)
    rows = []
    for m in models:
        leads = dl[f"lead_{m}"].to_numpy(float)
        row = {"modèle": m,
               "échéance_médiane_h": round(float(np.median(leads)), 1),
               "échéance_min_h": round(float(leads.min()), 1),
               "échéance_max_h": round(float(leads.max()), 1),
               "P10–P90_h": f"{np.percentile(leads, 10):.0f}–{np.percentile(leads, 90):.0f}",
               "n_jours": int(len(leads))}
        if df is not None and m in df.columns:
            j = df[["date", "tmax_c", m]].merge(dl[["date", f"lead_{m}"]], on="date", how="inner")
            if len(j) >= 60:
                err = np.abs(j[m].to_numpy(float) - j["tmax_c"].to_numpy(float))
                ld = j[f"lead_{m}"].to_numpy(float)
                med = float(np.median(ld))
                short, long_ = err[ld <= med], err[ld > med]
                if len(short) >= 20 and len(long_) >= 20:
                    row["MAE_éch.courte_°C"] = round(float(np.mean(short)), 3)
                    row["MAE_éch.longue_°C"] = round(float(np.mean(long_)), 3)
                    row["Δ_MAE_°C"] = round(float(np.mean(long_) - np.mean(short)), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("échéance_médiane_h").reset_index(drop=True)


def runs_table(models: list[str], peak_hour_local: int, sample_date, tz: str) -> pd.DataFrame:
    """Table par modèle : cadence, run utilisé par previous_day1 au pic (heure locale
    `peak_hour_local` du jour `sample_date`), échéance réelle min/typique/max."""
    d = pd.Timestamp(sample_date).to_pydatetime().replace(hour=int(peak_hour_local), minute=0)
    rows = []
    for m in models:
        info = run_for_valid_hour(m, d, tz)
        ih = MODEL_RUNS.get(m, {}).get("init_hours", _DEFAULT_INIT_HOURS)
        grid = ("horaire" if len(ih) == 24 else "/".join(f"{h:02d}" for h in sorted(ih)) + " UTC")
        rows.append({
            "modèle": m,
            "runs_UTC": grid,
            "run_utilisé_(pic)": info["init_utc"].strftime("%d/%m %H:%M UTC"),
            "échéance_au_pic_h": round(info["lead_h"], 1),
            "échéance_min–max_h": f"24 – {24 + _cadence_of(ih):.0f}",
            "note": MODEL_RUNS.get(m, {}).get("note", "grille inconnue → 00/06/12/18 supposée"),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- #
#  ⛰️  Heure du pic quotidien — depuis la STATION OFFICIELLE (METAR, archive IEM)
#
#  Polymarket règle sur les relevés de la station (Wunderground affiche ces mêmes
#  METAR) : l'heure du pic doit donc venir des OBSERVATIONS de la station, pas
#  d'une réanalyse sur grille. Source primaire : archive ASOS/METAR de l'Iowa
#  Environmental Mesonet (mesonet.agron.iastate.edu), interrogée par code ICAO,
#  heures RENDUES EN HEURE LOCALE par l'API (paramètre tz). Repli ERA5
#  (archive Open-Meteo à la position de la station) uniquement si la station
#  est indisponible (pas d'ICAO, ou archive vide).
# ----------------------------------------------------------------------------- #
def _peaks_from_hourly(h: pd.DataFrame) -> pd.DataFrame:
    """h : colonnes time (locale) + t. → une ligne par jour : heure du max."""
    h = h.dropna(subset=["t"]).copy()
    h["date"] = h["time"].dt.normalize()
    idx = h.groupby("date")["t"].idxmax()
    out = h.loc[idx, ["date", "time", "t"]].copy()
    out["peak_hour"] = out["time"].dt.hour
    return out[["date", "peak_hour", "t"]].rename(columns={"t": "tmax_h"}).reset_index(drop=True)


def _fetch_station_hourly(icao: str, tz: str, start: str, end: str) -> pd.DataFrame:
    """Relevés de température de la station (METAR horaires + spéciaux) via l'archive
    IEM ASOS. `valid` est déjà en heure locale (tz passé à l'API)."""
    import io
    d1, d2 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
           f"?station={icao}&data=tmpc"
           f"&year1={d1.year}&month1={d1.month}&day1={d1.day}"
           f"&year2={d2.year}&month2={d2.month}&day2={d2.day}"
           f"&tz={tz}&format=onlycomma&latlon=no&missing=M&trace=T"
           "&report_type=3&report_type=4")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "tmpc" not in df.columns or not len(df):
        raise RuntimeError(f"archive IEM vide pour {icao}")
    df = df[df["tmpc"] != "M"].copy()
    if not len(df):
        raise RuntimeError(f"aucune température IEM pour {icao}")
    return pd.DataFrame({"time": pd.to_datetime(df["valid"]),
                         "t": df["tmpc"].astype(float)})


def _fetch_era5_hourly(lat: float, lon: float, tz: str, start: str, end: str) -> pd.DataFrame:
    """Repli : température horaire ERA5 (archive Open-Meteo) à la position de la station."""
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&hourly=temperature_2m"
           f"&timezone={tz}&start_date={start}&end_date={end}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    j = r.json()["hourly"]
    return pd.DataFrame({"time": pd.to_datetime(j["time"]), "t": j["temperature_2m"]})


def fetch_peak_hours(slug: str, lat: float, lon: float, tz: str,
                     start: str, end: str, icao: str | None = None,
                     force: bool = False) -> pd.DataFrame:
    """Heure LOCALE du max quotidien de température, par jour. Source primaire :
    la station officielle (METAR via archive IEM, celle du règlement) ; repli ERA5.
    Cache JSON dans data/peak_hours/ (la source utilisée y est enregistrée et
    exposée via df.attrs["source"])."""
    PEAK_DIR.mkdir(parents=True, exist_ok=True)
    cache = PEAK_DIR / f"{slug}.json"
    if cache.exists() and not force:
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("start") <= start and d.get("end") >= end:
                df = pd.DataFrame(d["rows"])
                df["date"] = pd.to_datetime(df["date"])
                df.attrs["source"] = d.get("source", "?")
                return df
        except Exception:
            pass
    h, source = None, None
    if icao:
        try:
            h = _fetch_station_hourly(icao, tz, start, end)
            source = f"station {icao} (METAR, archive IEM)"
        except Exception:
            h = None
    if h is None:
        h = _fetch_era5_hourly(lat, lon, tz, start, end)
        source = "ERA5 (repli — station indisponible)" if icao else "ERA5 (pas d'ICAO)"
    out = _peaks_from_hourly(h)
    out.attrs["source"] = source
    cache.write_text(json.dumps({
        "start": start, "end": end, "source": source,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "rows": [{"date": str(r_.date.date()), "peak_hour": int(r_.peak_hour),
                  "tmax_h": float(r_.tmax_h)} for r_ in out.itertuples()],
    }, ensure_ascii=False), encoding="utf-8")
    return out


def peak_summary(peaks: pd.DataFrame) -> dict:
    """Médiane / P10–P90 de l'heure du pic (heure locale)."""
    ph = peaks["peak_hour"].to_numpy(int)
    return {"médiane": int(np.median(ph)),
            "p10": int(np.percentile(ph, 10)), "p90": int(np.percentile(ph, 90)),
            "n_jours": int(len(ph))}


# ----------------------------------------------------------------------------- #
#  🏆  Vue multiville : meilleur modèle par ville
# ----------------------------------------------------------------------------- #
def best_by_city(obs: pd.DataFrame, fc: pd.DataFrame, meta: dict,
                 metric: str = "MAE") -> pd.DataFrame:
    """Une ligne par ville : modèle retenu + sa MAE/RMSE, et l'écart avec le 2e."""
    col = "MAE_°C" if metric == "MAE" else "RMSE_°C"
    rows = []
    for slug in sorted(meta.keys()):
        models = models_of(meta, slug, only_used=True)
        if not models:
            continue
        df = city_frame(obs, fc, slug, models)
        if len(df) < 30:
            continue
        sc = model_scores(df, models).sort_values(col).reset_index(drop=True)
        best = sc.iloc[0]
        second = sc.iloc[1] if len(sc) > 1 else None
        rows.append({
            "ville": meta[slug]["name"], "slug": slug,
            "meilleur_modèle": best["modèle"],
            "MAE_°C": round(float(best["MAE_°C"]), 3),
            "RMSE_°C": round(float(best["RMSE_°C"]), 3),
            "biais_°C": round(float(best["biais_°C"]), 3),
            "2e_modèle": (second["modèle"] if second is not None else ""),
            f"écart_{metric}_vs_2e": (round(float(second[col] - best[col]), 3)
                                      if second is not None else None),
            "n_jours": int(best["n_jours"]),
        })
    return pd.DataFrame(rows)


def earliest_scan_time(model: str, peak_hour_local: int, sample_date, tz: str) -> dict:
    """V9 — À partir de quelle heure LOCALE (la veille de la cible) le scan est sûr.

    Le scan lit previous_day1 pour DEMAIN : la valeur au pic de demain vient du dernier
    run initialisé ≤ (pic_demain − 24 h), c'est-à-dire aujourd'hui vers l'heure du pic
    (en UTC). Cette valeur n'est FINALE qu'une fois ce run calculé, diffusé et ingéré
    (~4–6 h pour un global, ~1–3 h pour un régional). Scanner avant = lire une valeur
    provisoire du run précédent (échéance plus longue que la calibration, et qui
    changera après le scan). Scanner après ne change plus rien : la valeur est figée.

    Renvoie {run_init_utc, run_init_local, dispo_locale, delay_h} où dispo_locale est
    l'heure locale (la veille de sample_date) à partir de laquelle scanner est sûr."""
    target = pd.Timestamp(sample_date).to_pydatetime().replace(hour=int(peak_hour_local), minute=0)
    info = run_for_valid_hour(model, target, tz)          # run du pic de DEMAIN
    delay = int(_AVAIL_DELAY_H.get(model, 6))
    init_utc = info["init_utc"]
    avail_utc = init_utc + timedelta(hours=delay)
    loc = ZoneInfo(tz)
    return {"run_init_utc": init_utc,
            "run_init_local": init_utc.astimezone(loc),
            "dispo_locale": avail_utc.astimezone(loc),
            "delay_h": delay,
            "lead_h": info["lead_h"]}


def paired_spread_test(obs, fc, meta, slug: str, mode_a: str, mode_b: str,
                       reg_cfg=None) -> dict:
    """V9 — Test APPARIÉ de deux définitions du spread sur une ville : même μ (BEST),
    même famille de loi (Student-t, pour isoler l'effet du σ), CRPS calculé jour par
    jour sur le split de test, différence appariée → gain moyen, erreur-type, t.

    |t| > 2 ≈ gain réel ; |t| < 2 → indistinguable du bruit d'échantillonnage."""
    from .engine import (PipelineConfig, RegConfig, fit_regression, spread_series,
                         fit_sigma, sigma_of, _fit_family, _design)
    models = models_of(meta, slug, only_used=True)
    df = city_frame(obs, fc, slug, models)
    cut = int(len(df) * 0.7)
    tr, te = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    if len(te) < 40:
        return {"ok": False}
    cfg = PipelineConfig(reg=(reg_cfg or RegConfig(method="BEST")))
    reg = fit_regression(tr, models, cfg.reg)
    y = te["tmax_c"].to_numpy(float)
    Ate, _ = _design(te[models].to_numpy(float), te["doy"].to_numpy(), reg["use_season"])
    mu = Ate @ reg["beta_vector"]

    def crps_days(mode):
        sp_tr = spread_series(tr, models, mode)
        a, b = fit_sigma(sp_tr, np.abs(reg["resid"]), cfg.sigma)
        z_tr = reg["resid"] / sigma_of(sp_tr, a, b, cfg.sigma)
        d, _ = _fit_family(z_tr, "Student-t")
        sig = sigma_of(spread_series(te, models, mode), a, b, cfg.sigma)
        # grille vectorisée : q ∈ [-8, 8] en z, convertie par jour
        q = np.linspace(-8.0, 8.0, 481)
        Fz = d.cdf(q)                                     # (481,)
        X = mu[:, None] + sig[:, None] * q[None, :]       # (n, 481)
        H = (X >= y[:, None]).astype(float)
        integ = (Fz[None, :] - H) ** 2
        return np.trapezoid(integ, X, axis=1)

    ca, cb = crps_days(mode_a), crps_days(mode_b)
    diff = ca - cb                                        # >0 : mode_b meilleur
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    return {"ok": True, "crps_a": float(ca.mean()), "crps_b": float(cb.mean()),
            "gain_b": float(diff.mean()), "se": se,
            "t": float(diff.mean() / se) if se > 0 else 0.0, "n": int(len(diff))}


# ----------------------------------------------------------------------------- #
#  🕐  V10 — Cohérence d'échéance : stratification du socle
# ----------------------------------------------------------------------------- #
def lead_coherent_frame(df: pd.DataFrame, slug: str, spec: dict, model: str,
                        target_date, target_peak_hour: int | None,
                        tol: float = 1.0, min_rows: int = 80) -> tuple[pd.DataFrame, dict]:
    """Filtre `df` (city_frame) pour ne garder que les jours dont l'échéance
    previous_day1 (heure du pic OBSERVÉE à la station − run de la grille) est proche
    de l'échéance ATTENDUE demain (heure du pic PRÉVUE − run de la grille).

    target_peak_hour : pic prévu de demain (heure locale) — argmax de la série live ;
    si None, repli sur le pic médian observé. Tolérance ±tol h, élargie (+1 h à la
    fois) tant que la strate < min_rows ; au pire, repli poolé (df complet).

    Renvoie (df_filtré, info) où info = {applique, lead_cible_h, pic_utilisé,
    pic_source, tol_h, n_strate, n_total, run_init_utc}."""
    n_total = len(df)
    info = {"applique": False, "n_total": n_total, "n_strate": n_total,
            "lead_cible_h": None, "pic_utilisé": None, "pic_source": None,
            "tol_h": None, "run_init_utc": None}
    try:
        pk = fetch_peak_hours(slug, spec["lat"], spec["lon"], spec["tz"],
                              str(spec.get("date_min", "2024-06-01")),
                              str(pd.Timestamp(df["date"].max()).date()),
                              icao=spec.get("icao"))
    except Exception:
        return df, {**info, "pic_source": "⚠️ station indisponible → poolé"}
    if target_peak_hour is None:
        target_peak_hour = int(np.median(pk["peak_hour"]))
        info["pic_source"] = "médiane observée (pic prévu indisponible)"
    else:
        info["pic_source"] = "pic prévu (série live)"
    info["pic_utilisé"] = int(target_peak_hour)
    tgt = pd.Timestamp(target_date).to_pydatetime().replace(hour=int(target_peak_hour), minute=0)
    r = run_for_valid_hour(model, tgt, spec["tz"])
    info["lead_cible_h"] = round(float(r["lead_h"]), 1)
    info["run_init_utc"] = r["init_utc"].strftime("%d/%m %H:%M UTC")
    dl = daily_leads([model], pk, spec["tz"]).rename(columns={f"lead_{model}": "lead_h"})
    j = df.merge(dl[["date", "lead_h"]], on="date", how="left")
    known = j["lead_h"].notna()
    t = float(tol)
    while t < 24:
        m = known & ((j["lead_h"] - r["lead_h"]).abs() <= t)
        if int(m.sum()) >= min_rows:
            info.update({"applique": True, "tol_h": t, "n_strate": int(m.sum())})
            return df.loc[m.to_numpy()].reset_index(drop=True), info
        t += 1.0
    return df, {**info, "pic_source": info["pic_source"] + " → strate trop petite, poolé"}
