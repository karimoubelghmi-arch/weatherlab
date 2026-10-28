# -*- coding: utf-8 -*-
"""V14.1 — Nettoyage CHIRURGICAL des observations corrompues du socle.

Découverte (26/07/2026) : les pires « résidus » du socle ne sont pas des ratés
météo mais des OBSERVATIONS corrompues, concentrées sur les dates de CHANGEMENT
D'HEURE (Tmax calculée sur une fenêtre décalée/tronquée) :
    Denver −1,7 °C le 2025-03-09 (voisins 10 et 21 °C, prévisions ~18) ;
    Paris 6 °C / Madrid 9 °C le 2025-03-30 ; Chicago 1,1 °C le 2026-03-08 ; etc.

Règle de détection (déterministe, mesurée sur les 49 villes → ~14 jours) :
un jour est corrompu si, dans la table complète de la ville,
    1. l'observation ROMPT avec ses deux voisines : min(|Δ veille|, |Δ lendemain|) ≥ 7 °C ;
    2. elle contredit les modèles : |obs − moyenne des modèles| ≥ 7 °C ;
    3. les modèles, eux, sont COHÉRENTS avec les jours voisins :
       |obs − moy. modèles| < 4 °C la veille ET le lendemain.
La condition 3 est ce qui distingue une donnée corrompue d'un vrai extrême
météo (brouillard de Lucknow, coups de froid) : lors d'un vrai extrême, les
jours voisins montrent aussi la perturbation ou les modèles se trompent aussi
autour — on GARDE alors le jour (les vrais extrêmes nourrissent les queues des
lois ; l'épuration statistique |z|>seuil a été testée et DÉGRADE la couverture,
voir LISEZMOI_V14.1).

Usage : appelé automatiquement par load_socle(). Chaque exclusion est
journalisée dans data/socle_exclusions.csv (relisible, re-générée à chaque
chargement). Pour forcer la CONSERVATION d'un jour détecté à tort, ajouter
"slug:AAAA-MM-JJ" à la liste JSON data/socle_keep.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXCLUSIONS_CSV = DATA_DIR / "socle_exclusions.csv"
KEEP_PATH = DATA_DIR / "socle_keep.json"
DROP_PATH = DATA_DIR / "socle_drop.json"

# seuils de la règle (ceux validés sur les 49 villes)
JUMP_MIN_C = 7.0      # rupture minimale avec les DEUX voisins
DISAGREE_MIN_C = 7.0  # désaccord minimal obs ↔ moyenne des modèles le jour même
NEIGHB_OK_C = 4.0     # accord obs ↔ modèles exigé la veille et le lendemain


def _keep_map() -> dict:
    """Liste blanche : {"slug:AAAA-MM-JJ": "raison de la conservation"}.
    (Une simple liste JSON est aussi acceptée.)"""
    try:
        raw = json.loads(KEEP_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        return {str(x): "liste blanche" for x in raw}
    except Exception:
        return {}


def _drop_map() -> dict:
    """Liste noire manuelle : {"slug:AAAA-MM-JJ": "raison"} — jours corrompus
    VÉRIFIÉS à la main (archives réelles) que la règle automatique ne voit pas."""
    try:
        raw = json.loads(DROP_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        return {str(x): "liste noire manuelle" for x in raw}
    except Exception:
        return {}


def find_corrupt_obs(obs: pd.DataFrame, fc: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Détecte les observations corrompues sur tout le socle.

    Renvoie un DataFrame (slug, date, obs_°C, prévision_moy_°C, saut_voisins_°C,
    raison) — vide si rien. N'écrit rien : pur calcul."""
    from .data import city_frame, models_of

    rows = []
    kept = _keep_map()
    for slug in sorted(meta):
        models = models_of(meta, slug, only_used=True)
        if not models:
            continue
        df = city_frame(obs, fc, slug, models).reset_index(drop=True)
        if len(df) < 10:
            continue
        t = df["tmax_c"].to_numpy(float)
        pm = df[models].mean(axis=1).to_numpy(float)
        for i in range(1, len(df) - 1):
            jump = min(abs(t[i] - t[i - 1]), abs(t[i] - t[i + 1]))
            if (jump >= JUMP_MIN_C
                    and abs(t[i] - pm[i]) >= DISAGREE_MIN_C
                    and abs(t[i - 1] - pm[i - 1]) < NEIGHB_OK_C
                    and abs(t[i + 1] - pm[i + 1]) < NEIGHB_OK_C):
                d = str(pd.Timestamp(df.iloc[i]["date"]).date())
                key = f"{slug}:{d}"
                rows.append({"slug": slug, "date": d,
                             "obs_°C": round(float(t[i]), 2),
                             "prévision_moy_°C": round(float(pm[i]), 2),
                             "saut_voisins_°C": round(float(jump), 2),
                             "action": ("conservé (liste blanche)" if key in kept else "exclu"),
                             "raison": (kept.get(key) or
                                        "rupture obs vs voisins, modèles cohérents (type changement d'heure)")})
    # liste noire manuelle : toujours exclue, journalisée aussi
    for key, why in _drop_map().items():
        try:
            slug, d = key.split(":", 1)
        except ValueError:
            continue
        if any(r["slug"] == slug and r["date"] == d for r in rows):
            continue
        o = obs[(obs["slug"] == slug) & (obs["date"] == pd.Timestamp(d))]
        rows.append({"slug": slug, "date": d,
                     "obs_°C": (round(float(o["tmax_c"].iloc[0]), 2) if len(o) else np.nan),
                     "prévision_moy_°C": np.nan, "saut_voisins_°C": np.nan,
                     "action": "exclu", "raison": f"liste noire manuelle : {why}"})
    return pd.DataFrame(rows, columns=["slug", "date", "obs_°C", "prévision_moy_°C",
                                       "saut_voisins_°C", "action", "raison"])


def clean_obs(obs: pd.DataFrame, fc: pd.DataFrame, meta: dict,
              write_log: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retire du socle en mémoire les observations corrompues détectées.

    Renvoie (obs_nettoyées, exclusions). Le parquet sur disque n'est JAMAIS
    modifié (nettoyage non destructif, refait à chaque chargement) ; la liste
    des jours écartés est journalisée dans data/socle_exclusions.csv."""
    det = find_corrupt_obs(obs, fc, meta)
    if write_log:
        try:
            det.to_csv(EXCLUSIONS_CSV, index=False)
        except Exception:
            pass
    exc = det[det["action"] == "exclu"] if len(det) else det
    if not len(exc):
        return obs, det
    bad = set(zip(exc["slug"], pd.to_datetime(exc["date"])))
    mask = [(s, d) in bad for s, d in zip(obs["slug"], obs["date"])]
    return obs[~np.asarray(mask)].reset_index(drop=True), det
