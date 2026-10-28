"""Récupération LIVE des prévisions Open-Meteo (previous-runs, J-1, lead ~24 h).

Porté depuis meteo_bot.prevday1_max : pour une date cible et un modèle, on lit le champ
`temperature_2m_previous_day1` (la prévision telle qu'elle était la veille) et on prend
le max journalier en heure locale. C'est EXACTEMENT la variable sur laquelle la chaîne
a été calibrée. Valeurs renvoyées en °C (le moteur travaille en °C en interne).

Nécessite un accès réseau. Sans clé API.
"""
from __future__ import annotations

import json
import urllib.parse

from .net import http_get

BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"


def forecast_source() -> str:
    """V18 : source FIGEE sur 'previous_day1', la variable de calibration.
    Le live lit exactement ce que l'entrainement a vu (scanne en soiree locale,
    quand le pic du lendemain est fige). L'ancien mode 'courante' (dernier run)
    est retire : il injectait des previsions fraiches dans des beta/sigma appris
    sur des vieilles -- incoherence corrigee par cette version."""
    return "previous_day1"


def local_today(tz: str):
    """Date courante DANS le fuseau de la ville (pas celui de la machine)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(tz)).date()


def local_target_date(tz: str, offset_days: int = 1):
    """Date locale de la ville + offset (1 = 'demain heure locale')."""
    from datetime import timedelta
    return local_today(tz) + timedelta(days=offset_days)


def fetch_model_tmax_peak(model: str, lat: float, lon: float, tz: str, date):
    """(Tmax_°C, heure_locale_du_pic_PRÉVU) J-1 par UN modèle pour `date`.
    V10 : le pic prévu (argmax de la série horaire previous_day1) sert au mode
    « cohérence d'échéance » — aucune requête supplémentaire. (None, None) si indispo."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(date), "end_date": str(date),
        "hourly": "temperature_2m_previous_day1",
        "timezone": tz, "models": model,
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    h = json.loads(http_get(url)).get("hourly", {})
    pairs = [(t, v) for t, v in zip(h.get("time", []), h.get("temperature_2m_previous_day1", []))
             if v is not None and str(t)[:10] == str(date)]
    if not pairs:
        return None, None
    t_best, v_best = max(pairs, key=lambda tv: tv[1])
    try:
        peak_h = int(str(t_best)[11:13])
    except Exception:
        peak_h = None
    return v_best, peak_h


def fetch_model_tmax(model: str, lat: float, lon: float, tz: str, date) -> float | None:
    """Tmax (°C) prévue J-1 par UN modèle pour `date` (YYYY-MM-DD). None si indisponible."""
    return fetch_model_tmax_peak(model, lat, lon, tz, date)[0]


def fetch_model_tmax_current(model: str, lat: float, lon: float, tz: str, date) -> float | None:
    """Tmax (°C) du DERNIER run disponible pour `date`. None si indisponible."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(date), "end_date": str(date),
        "hourly": "temperature_2m",
        "timezone": tz, "models": model,
    }
    url = FORECAST_BASE + "?" + urllib.parse.urlencode(params)
    h = json.loads(http_get(url)).get("hourly", {})
    vals = [v for t, v in zip(h.get("time", []), h.get("temperature_2m", []))
            if v is not None and str(t)[:10] == str(date)]
    return max(vals) if vals else None


def fetch_forecasts(models: list[str], lat: float, lon: float, tz: str, date,
                    source: str | None = None) -> dict:
    """Renvoie {modèle: Tmax_°C ou None} pour la date cible. Une requête par modèle.

    source 'courante' : dernier run (echeance plus courte -> facteur sigma applique
    par le scanner). source 'previous_day1' : variable historique de calibration.
    En mode courant, un modèle indisponible retombe silencieusement sur previous_day1."""
    src_ = source or forecast_source()
    out = {}
    for m in models:
        try:
            if src_ == "courante":
                v = fetch_model_tmax_current(m, lat, lon, tz, date)
                if v is None:
                    v = fetch_model_tmax(m, lat, lon, tz, date)
                out[m] = v
            else:
                out[m] = fetch_model_tmax(m, lat, lon, tz, date)
        except Exception:
            out[m] = None
    return out


def fetch_forecasts_with_peaks(models: list[str], lat: float, lon: float, tz: str, date) -> tuple[dict, dict]:
    """V10 — comme fetch_forecasts (source previous_day1) mais renvoie AUSSI
    {modèle: heure_locale_du_pic_prévu}. Une requête par modèle, inchangé."""
    vals, peaks = {}, {}
    for m in models:
        try:
            v, ph = fetch_model_tmax_peak(m, lat, lon, tz, date)
            vals[m], peaks[m] = v, ph
        except Exception:
            vals[m], peaks[m] = None, None
    return vals, peaks
