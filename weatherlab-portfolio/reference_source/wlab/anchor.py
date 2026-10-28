# -*- coding: utf-8 -*-
"""V13 — Ancre J-2 : deuxième calibration complète à l'horizon 48 h.

Motivation (mesure empirique `mesure_double_comptage.py`, 35 villes) :
  - la loi de puissance RMSE·(L/24)^α appliquée par-dessus σ = a₁+b₁·spread
    compte partiellement DEUX FOIS l'élargissement (le spread grandit déjà
    avec l'horizon) : biais net ≈ +9 % en moyenne mais de −38 % à +35 %
    selon la ville — indétectable par l'architecture v9-v12 ;
  - les β appris à 24 h ne sont pas les bons poids à 48 h : refitter β₂ sur
    les archives previous_day2 réduit l'erreur μ de ~14 % en moyenne.

Principe : `calibrate_anchor_j2.py` calibre par ville une VRAIE ancre 48 h
(β₂ refités + a₂, b₂ rétrécis vers l'ancre 24 h) → data/anchor_j2.json.
À l'exécution, pour une échéance L ∈ [24 h, 48 h] :

    R(s)     = σ₂(s) / σ₁(s)          (ratio CONDITIONNEL au spread du jour)
    scale(L) = R(s) ^ ((L−24)/24)      (interpolation géométrique 24→48 h)
    μ(L)     = (1−w)·μ₁ + w·μ₂,  w = (L−24)/24

Au-delà de 48 h : scale(L) = R(s) × (L/48)^α (prolongement par l'ancienne loi
de puissance, α de la ville ou 0,28). Bornes de sécurité [0.75, 1.60].
Sous 24 h et villes sans ancre : comportement v9 inchangé (grille + défaut).
"""
from __future__ import annotations

import json

import numpy as np

from .data import DATA_DIR

ANCHOR_PATH = DATA_DIR / "anchor_j2.json"
SCALE_MIN, SCALE_MAX = 0.75, 1.60      # bornes du facteur ancré
RATIO_MIN, RATIO_MAX = 0.60, 1.80      # bornes du ratio brut σ₂/σ₁
ALPHA_TAIL = 0.28                       # prolongement > 48 h si α ville inconnu
MU_GUARD_C = 3.0                        # garde-fou : |μ₂−μ₁| max accepté (°C)
# V15 — PÉREMPTION : l'ancre est ajustée sur une fenêtre GLISSANTE de 92 jours
# d'archives previous_day2. Ses β₂/a₂/b₂ décrivent donc la saison de cette
# fenêtre, pas la météo « en général ». Au-delà de MAX_AGE_DAYS sans
# recalibration, l'ancre est refusée (repli v9 : loi de puissance + défaut
# global) plutôt que d'appliquer des coefficients d'une autre saison.
MAX_AGE_DAYS = 45

_cache: dict = {}


def _table() -> dict:
    if "t" not in _cache:
        try:
            _cache["t"] = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
        except Exception:
            _cache["t"] = {}
    return _cache["t"]


def reload() -> None:
    _cache.pop("t", None)


def age_days() -> float | None:
    """Âge du fichier d'ancres en jours (via _meta.generated_utc, sinon mtime)."""
    from datetime import datetime, timezone
    meta = _table().get("_meta")
    if isinstance(meta, dict) and meta.get("generated_utc"):
        try:
            dt = datetime.fromisoformat(str(meta["generated_utc"]))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        except Exception:
            pass
    try:
        mt = ANCHOR_PATH.stat().st_mtime
        return (datetime.now(timezone.utc).timestamp() - mt) / 86400.0
    except Exception:
        return None


def health() -> dict:
    """État du fichier d'ancres : âge, péremption, ancres au format empoisonné.

    Format « empoisonné » = β₂ ajustés AVEC sin/cos sur ~50 jours (bug V13-V14 :
    inidentifiables, ils explosent hors saison — cf. LISEZMOI_V14.1). Ces ancres
    sont refusées par get() même si le fichier est récent.
    """
    t = _table()
    poisoned = [k for k, v in t.items()
                if isinstance(v, dict) and v.get("flag") == "ok"
                and (v.get("reg") or {}).get("use_season")]
    n_ok = sum(1 for k, v in t.items()
               if isinstance(v, dict) and v.get("flag") == "ok" and k not in poisoned)
    age = age_days()
    return {"n_ok": n_ok, "poisoned": poisoned, "age_days": age,
            "stale": (age is None or age > MAX_AGE_DAYS),
            "max_age_days": MAX_AGE_DAYS}


def get(slug: str) -> dict | None:
    """Ancre de la ville, ou None si non calibrée / invalide / périmée.

    V15 — deux refus supplémentaires (repli v9 propre dans les deux cas) :
      - format empoisonné : reg.use_season vrai (β₂ avec sin/cos, bug V13-V14) ;
      - fichier périmé : > MAX_AGE_DAYS depuis la calibration (fenêtre 92 j
        glissante → les coefficients suivent la saison, ils ne sont pas éternels).
    """
    e = _table().get(slug)
    if not isinstance(e, dict) or e.get("flag") != "ok":
        return None
    if not all(k in e for k in ("a2", "b2", "reg")):
        return None
    if (e.get("reg") or {}).get("use_season"):
        return None                     # ancre empoisonnée (sin/cos) — jamais utilisée
    age = age_days()
    if age is None or age > MAX_AGE_DAYS:
        return None                     # ancre périmée — recalibrer (calibrate_anchor_j2.py)
    return e


def adjust(anchor: dict, lead_h: float, mu1: float, spread_t: float,
           a1: float, b1: float, sigma_cfg, target_forecasts: dict,
           target_doy: int, alpha_tail: float | None = None) -> dict:
    """Facteur d'échéance et μ ancrés pour une échéance > 24 h.

    Renvoie {"scale", "mu", "ratio", "w", "mu_blend"(bool)}.
    Ne s'applique JAMAIS sous 24 h (l'appelant garde la grille v9).
    """
    from .engine import sigma_of, predict_mu

    lead_h = float(lead_h)
    w = min(1.0, max(0.0, (lead_h - 24.0) / 24.0))

    # --- ratio conditionnel au spread DU JOUR ---
    s1 = float(sigma_of(spread_t, float(a1), float(b1), sigma_cfg))
    s2 = float(sigma_of(spread_t, float(anchor["a2"]), float(anchor["b2"]), sigma_cfg))
    ratio = s2 / s1 if s1 > 1e-9 else 1.0
    ratio = min(RATIO_MAX, max(RATIO_MIN, ratio))

    scale = ratio ** w
    if lead_h > 48.0:
        at = float(alpha_tail) if isinstance(alpha_tail, (int, float)) and alpha_tail > 0 else ALPHA_TAIL
        scale *= (lead_h / 48.0) ** at
    scale = min(SCALE_MAX, max(SCALE_MIN, scale))

    # --- μ : mélange linéaire vers β₂ si les modèles de l'ancre sont dispo ---
    # ⚠️ GARDE-FOU (26/07/2026) : β₂ est ajusté sur ~50 jours consécutifs AVEC des
    # termes de saison sin/cos — inidentifiables sur une fenêtre de 7 semaines, ils
    # explosent HORS de la saison de calibration (mesuré : Séoul, prévisions à 31 °C
    # → μ₂ = 8,8 °C avec sin=+18.8, cos=−24.4 !). Si μ₂ s'écarte de μ₁ de plus de
    # MU_GUARD_C, l'ancre μ est REFUSÉE (on garde μ₁) et la raison est exposée.
    # Recalibrer avec `calibrate_anchor_j2.py` (β₂ désormais SANS saison) purge le mal.
    mu, blended, guard = float(mu1), False, None
    reg2 = anchor.get("reg") or {}
    m2 = reg2.get("models") or []
    if w > 0 and m2 and all(m in target_forecasts for m in m2):
        try:
            mu2 = float(predict_mu(reg2, {m: float(target_forecasts[m]) for m in m2},
                                   int(target_doy)))
            if abs(mu2 - float(mu1)) > MU_GUARD_C:
                guard = f"μ₂={mu2:.1f} vs μ₁={float(mu1):.1f} (>±{MU_GUARD_C:g} °C) — ancre μ refusée"
            else:
                mu = (1.0 - w) * float(mu1) + w * mu2
                blended = True
        except Exception:
            pass

    return {"scale": float(scale), "mu": mu, "ratio": round(ratio, 3),
            "w": round(w, 3), "mu_blend": blended, "garde_μ": guard}
