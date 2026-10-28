# -*- coding: utf-8 -*-
"""Multiplicateur de sigma selon l'age de l'information (v2, symetrique).

Reference : echeance 24 h = variable de calibration -> facteur 1.0.
  < 24 h : info plus fraiche -> facteur < 1 (interpolation grille 6/12/18/24,
           extrapolation sous 24 h, plancher 0.70).
  > 24 h : info plus vieille -> facteur > 1 via la loi (L/24)^alpha, mesuree
           sur les points 48/72/120/168 h de la calibration, plafond 1.40.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# v9.1 — reference = HEURE DE RECUPERATION DES PREVISIONS, sans latence de run.
#
# Avant : RUN_LATENCY_H = 3.0, ajoute a (pic - maintenant) pour approcher l'age reel
# du run tenu. Retire, pour deux raisons :
#   1. asymetrie — le cote calibration (`previous_day1`) est etiquete 24 h tout rond et
#      n'ajoute AUCUNE latence ; ajouter 3 h uniquement a l'inference decale la reference
#      (facteur 1.0 obtenu a 18 h la veille au lieu de 15 h) ;
#   2. fausse valeur — l'age reel des runs mesure via l'API de metadonnees Open-Meteo
#      s'etale de 3 h (ICON-EU) a 12 h (ECMWF, JMA, UKMO, CMA) et oscille au fil de la
#      journee. Une constante ne peut pas representer ca.
# On mesure donc simplement le temps entre la RECUPERATION des previsions et le pic
# vise : imprecis, mais coherent des deux cotes et sans parametre arbitraire.
RUN_LATENCY_H = 0.0
PEAK_HOUR = 15
_GRID = [6.0, 12.0, 18.0, 24.0]
FLOOR = 0.70
CAP = 1.40
ALPHA_DEFAULT = 0.28   # mediane des villes mesurees, pour les villes en defaut global
_cache: dict = {}


def _table() -> dict:
    if "t" not in _cache:
        p = DATA_DIR / "sigma_lead_factors.json"
        try:
            _cache["t"] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            _cache["t"] = {}
    return _cache["t"]


def lead_hours(tz: str, target_date, now=None) -> float:
    """Heures entre `now` et le pic (15 h locales) de `target_date`.

    `now` = MOMENT DE RECUPERATION DES PREVISIONS (pas l'heure courante) quand on le
    connait : si tu recuperes a 20 h et que tu scannes a 22 h, l'information que tu
    tiens a bien l'age de 20 h. A defaut, l'heure courante dans le fuseau de la ville.
    """
    now = now or datetime.now(ZoneInfo(tz))
    peak = datetime(target_date.year, target_date.month, target_date.day,
                    PEAK_HOUR, 0, tzinfo=ZoneInfo(tz))
    return max(1.0, (peak - now).total_seconds() / 3600.0 + RUN_LATENCY_H)


def sigma_scale(slug: str, lead_h: float) -> float:
    entry = _table().get(slug) or {}
    if lead_h >= 24.0:
        # info PLUS VIEILLE que la calibration -> elargir sigma (zone mesuree)
        alpha = entry.get("alpha")
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            alpha = ALPHA_DEFAULT
        return float(min(CAP, (lead_h / 24.0) ** float(alpha)))
    # info PLUS FRAICHE -> retrecir sigma (grille calibree, extrapolation sous 24 h)
    fac = entry.get("factors") or {}
    pts = [(g, float(fac.get(str(int(g)), 1.0))) for g in _GRID]
    lead_h = max(lead_h, _GRID[0])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= lead_h <= x1:
            f = y0 + (y1 - y0) * (lead_h - x0) / (x1 - x0)
            return float(min(1.0, max(FLOOR, f)))
    return 1.0


# --------------------------------------------------------------------------- #
#  v9 : point d'entree UNIQUE utilise par l'app, le scanner et l'auto-trader.
#  Avant, chaque appelant re-implementait la condition « source == courante » et
#  certains (prevision ville unique / multiville) ne l'appliquaient pas du tout :
#  le facteur etait affiche mais jamais integre a la distribution.
# --------------------------------------------------------------------------- #
def fetch_moment(entry: dict | None):
    """Instant de RECUPERATION des previsions d'une ville, lu dans le cache live.

    Renvoie un datetime timezone-aware (UTC) ou None si l'entree n'en porte pas
    (cache d'une version anterieure) — dans ce cas l'appelant retombe sur l'heure
    courante, comme avant.
    """
    if not isinstance(entry, dict):
        return None
    raw = entry.get("fetched_ts")
    if not raw:
        return None
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(str(raw))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def scale_for(slug: str, tz: str, target_date, enabled: bool = True,
              source: str | None = None, now=None) -> dict:
    """Renvoie {'lead_h', 'scale', 'applique', 'raison'} pour une ville / date cible.

    Le facteur n'a de sens que si l'information est PLUS FRAICHE (ou plus vieille)
    que la variable de calibration `previous_day1` (~24 h). Si la source des
    previsions EST `previous_day1`, l'echeance est deja celle de la calibration et
    le facteur vaut 1.0 par construction.
    """
    if source is None:
        try:
            from .live import forecast_source
            source = forecast_source()
        except Exception:
            source = "courante"
    try:
        lh = lead_hours(tz, target_date, now=now)
    except Exception:
        return {"lead_h": None, "scale": 1.0, "applique": False, "raison": "echeance incalculable"}
    if not enabled:
        return {"lead_h": lh, "scale": 1.0, "applique": False, "raison": "desactive dans la calibration"}
    if source != "courante":
        return {"lead_h": lh, "scale": 1.0, "applique": False,
                "raison": "source previous_day1 = echeance de calibration (24 h)"}
    sc = float(sigma_scale(slug, lh))
    return {"lead_h": lh, "scale": sc, "applique": True,
            "reference": ("récupération des prévisions" if now is not None else "heure courante"),
            "raison": ("info plus fraiche que la calibration" if sc < 1.0
                       else "info plus vieille que la calibration" if sc > 1.0 else "echeance = 24 h")}
