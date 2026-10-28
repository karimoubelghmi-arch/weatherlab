"""Persistance des préférences utilisateur (calibration).

Tout est stocké dans data/user_prefs.json :
  - models_excluded : {slug: [modèles exclus]}  → sélection de modèles par ville
  - sidebar         : réglages globaux de calibration (méthode, spread, loi, critère…)

Le fichier est appliqué au chargement du socle (load_socle) : les exclusions
sauvegardées écrasent celles de metadata.json. metadata.json n'est jamais modifié.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PREFS_PATH = DATA_DIR / "user_prefs.json"


def load_prefs() -> dict:
    """Lit user_prefs.json ({} si absent ou illisible)."""
    if PREFS_PATH.exists():
        try:
            return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def apply_model_prefs(meta: dict, prefs: dict) -> None:
    """Applique les exclusions sauvegardées sur META (en place).

    Seules les villes présentes dans les prefs sont touchées ; pour elles,
    la liste sauvegardée fait foi (un modèle absent de la liste est ré-inclus).
    """
    for slug, excluded in (prefs.get("models_excluded") or {}).items():
        if slug not in meta:
            continue
        excluded = set(excluded)
        for m in meta[slug].get("models", []):
            m["excluded"] = m["name"] in excluded


def collect_model_prefs(meta: dict) -> dict:
    """Photographie l'état actuel des exclusions de META (toutes les villes)."""
    return {slug: [m["name"] for m in spec.get("models", []) if m.get("excluded")]
            for slug, spec in meta.items()}


def save_prefs(meta: dict, sidebar: dict) -> dict:
    """Écrit modèles par ville + réglages sidebar, en PRÉSERVANT les autres sections
    (trading, auto…) déjà présentes dans le fichier."""
    return update_prefs(models_excluded=collect_model_prefs(meta), sidebar=dict(sidebar))


def update_prefs(**sections) -> dict:
    """Met à jour une ou plusieurs sections de user_prefs.json SANS toucher aux autres.
    Ex. update_prefs(trading={...}) préserve models_excluded, sidebar, auto…"""
    prefs = load_prefs()
    for k, v in sections.items():
        prefs[k] = v
    PREFS_PATH.write_text(json.dumps(prefs, indent=2, ensure_ascii=False), encoding="utf-8")
    return prefs
