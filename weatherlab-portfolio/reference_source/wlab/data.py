"""Couche d'accès aux données — lit le socle (dataset/ + metadata.json).

Le reste du logiciel ne lit JAMAIS l'Excel : tout passe par ici.
Tout est servi en °C (colonnes *_c). L'unité de règlement reste disponible
dans metadata pour parser/afficher les buckets Polymarket.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _read(name: str) -> pd.DataFrame:
    """Parquet si possible (rapide), sinon CSV (universel)."""
    pq = DATA_DIR / f"{name}.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def load_socle():
    obs = _read("observations")
    fc = _read("forecasts")
    obs["date"] = pd.to_datetime(obs["date"])
    fc["date"] = pd.to_datetime(fc["date"])
    meta = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    # préférences utilisateur sauvegardées (sélection de modèles par ville) :
    # elles écrasent les flags "excluded" de metadata.json, qui reste intact.
    from .prefs import load_prefs, apply_model_prefs
    apply_model_prefs(meta, load_prefs())
    snap_path = DATA_DIR / "calibration_snapshot.json"
    snap = json.loads(snap_path.read_text(encoding="utf-8")) if snap_path.exists() else {}
    # V14.1 — nettoyage chirurgical : retire les observations CORROMPUES (type
    # changement d'heure : rupture avec les voisins alors que les modèles sont
    # cohérents). Non destructif (parquet intact), journalisé dans
    # data/socle_exclusions.csv, liste blanche possible via data/socle_keep.json.
    # Les VRAIS extrêmes météo sont conservés (voir wlab/clean.py).
    from .clean import clean_obs
    obs, _exclusions = clean_obs(obs, fc, meta)
    return obs, fc, meta, snap


def city_list(meta: dict) -> list[str]:
    return sorted(meta.keys())


def models_of(meta: dict, slug: str, only_used: bool = True) -> list[str]:
    ms = meta[slug]["models"]
    return [m["name"] for m in ms if (not only_used) or (not m["excluded"])]


def city_frame(obs: pd.DataFrame, fc: pd.DataFrame, slug: str, models: list[str]) -> pd.DataFrame:
    """Table large d'UNE ville : une ligne par date, colonnes des modèles choisis (°C),
    Tmax (°C), doy, n_obs. On ne garde que les lignes COMPLÈTES sur les modèles choisis."""
    o = obs[obs["slug"] == slug][["date", "doy", "tmax_c", "tmax_native", "n_obs"]].copy()
    f = fc[(fc["slug"] == slug) & (fc["model"].isin(models))]
    wide = f.pivot_table(index="date", columns="model", values="forecast_c", aggfunc="first")
    wide = wide.reindex(columns=models)
    df = o.merge(wide, on="date", how="left").sort_values("date").reset_index(drop=True)
    complete = df["tmax_c"].notna() & df[models].notna().all(axis=1)
    df = df[complete].reset_index(drop=True)
    # V11 : « erreur d'hier » = Tmax observée de la VEILLE − moyenne des modèles de la
    # VEILLE (les deux connues au scan du soir). 0 quand la veille manque (repli neutre).
    prev = df[["date", "tmax_c"] + models].copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=1)
    prev["err_prev"] = prev["tmax_c"] - prev[models].mean(axis=1)
    df = df.merge(prev[["date", "err_prev"]], on="date", how="left")
    df["err_prev"] = df["err_prev"].fillna(0.0)
    return df


def all_cities_overview(obs: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Vue 'toutes les villes' pour la page historique."""
    rows = []
    for slug, g in obs.groupby("slug"):
        spec = meta.get(slug, {})
        rows.append({
            "ville": spec.get("name", slug),
            "slug": slug,
            "groupe": spec.get("group", ""),
            "unité": spec.get("settlement_unit", ""),
            "station": spec.get("station", ""),
            "n_jours": int(g["date"].notna().sum()),
            "début": g["date"].min().date(),
            "fin": g["date"].max().date(),
            "Tmax_min_°C": round(float(g["tmax_c"].min()), 1),
            "Tmax_moy_°C": round(float(g["tmax_c"].mean()), 1),
            "Tmax_max_°C": round(float(g["tmax_c"].max()), 1),
            "modèles_utilisés": spec.get("n_models_used", ""),
        })
    return pd.DataFrame(rows).sort_values("ville").reset_index(drop=True)
