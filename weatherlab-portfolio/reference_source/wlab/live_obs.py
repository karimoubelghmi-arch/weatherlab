# -*- coding: utf-8 -*-
"""Observations station en quasi temps réel, multi-sources + pic prévu.

Hiérarchie de fraîcheur par ville (la meilleure source disponible gagne) :
  1. AMeDAS (JMA, Japon)      : pas de 10 min, sans clé, latence ~1-2 min
  2. Synoptic Data (US)       : pas de ~5 min, token gratuit requis
                                (variable d'env SYNOPTIC_TOKEN)
  3. METAR NOAA (mondial)     : 30 min intl / horaire+SPECI US, sans clé

+ pic prévu du jour et du lendemain via Open-Meteo (heure du max horaire
  et « plateau » = heures à moins de 0.3° du max).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

_UA = {"User-Agent": "WeatherLab/2.0 (observations temps réel)"}
_TIMEOUT = 25
_cache: dict = {}

# Stations AMeDAS par ICAO (extensible si tu ajoutes des villes JP)
AMEDAS_BY_ICAO = {"RJTT": "44166"}  # Haneda

SYNOPTIC_TOKEN = os.environ.get("SYNOPTIC_TOKEN", "").strip()

# --- TLS robuste -------------------------------------------------------------
# Trois causes possibles de CERTIFICATE_VERIFY_FAILED selon la machine :
#   (a) Python de python.org sur macOS non relié aux certificats système ;
#   (b) antivirus/VPN/proxy qui intercepte le HTTPS (cert auto-signé) ;
#   (c) bundle certifi absent.
# On essaie donc plusieurs contextes dans l'ordre et on mémorise celui qui
# marche : trousseau système via `truststore` (règle (a) ET (b) si le cert de
# l'intercepteur est dans le trousseau) -> défaut système -> certifi.
# Dernier recours explicite : WLAB_SSL_NOVERIFY=1 désactive la vérification.
import ssl

def _candidate_contexts():
    ctxs = []
    try:
        import truststore  # pip install truststore
        ctxs.append(("truststore (trousseau système)", truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)))
    except Exception:
        pass
    ctxs.append(("défaut système", ssl.create_default_context()))
    try:
        import certifi
        ctxs.append(("certifi", ssl.create_default_context(cafile=certifi.where())))
    except Exception:
        pass
    if os.environ.get("WLAB_SSL_NOVERIFY") == "1":
        c = ssl.create_default_context(); c.check_hostname = False; c.verify_mode = ssl.CERT_NONE
        ctxs.append(("NON VÉRIFIÉ (WLAB_SSL_NOVERIFY=1)", c))
    return ctxs

_working_ctx: list = []  # [(label, ctx)] une fois trouvé

def _urlopen(req):
    if _working_ctx:
        return urllib.request.urlopen(req, timeout=_TIMEOUT, context=_working_ctx[0][1])
    last = None
    for label, ctx in _candidate_contexts():
        try:
            r = urllib.request.urlopen(req, timeout=_TIMEOUT, context=ctx)
            _working_ctx.append((label, ctx))
            return r
        except urllib.error.URLError as e:
            last = e
            if not isinstance(getattr(e, "reason", None), ssl.SSLError):
                raise  # pas un problème TLS : inutile d'essayer d'autres contextes
    raise RuntimeError(
        "Impossible d'établir une connexion TLS (certificat non vérifiable). "
        "Un antivirus/VPN/proxy intercepte probablement le HTTPS. Solutions : "
        "1) pip3 install truststore  (utilise le trousseau macOS)  "
        "2) macOS/python.org : lancer 'Install Certificates.command' dans /Applications/Python 3.x/  "
        "3) dernier recours : variable d'env WLAB_SSL_NOVERIFY=1"
    ) from last


def _get(url: str, ttl: int, key=None):
    key = key or url
    hit = _cache.get(key)
    now = time.time()
    if hit and now - hit[0] < ttl:
        return hit[1]
    req = urllib.request.Request(url, headers=_UA)
    with _urlopen(req) as r:
        data = json.loads(r.read())
    _cache[key] = (now, data)
    return data


# ----------------------------------------------------------------- METAR ----
def fetch_metar(icao: str, hours: int = 48, ttl: int = 55) -> pd.DataFrame:
    url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=json&hours={int(hours)}"
    rows = []
    for m in _get(url, ttl):
        if m.get("temp") is None:
            continue
        raw = m.get("rawOb", "") or ""
        rows.append({
            "time_utc": pd.Timestamp(m["reportTime"], tz="UTC"),
            "temp_c": float(m["temp"]),
            "flag": "SPECI" if raw.startswith("SPECI") else "",
            "raw": raw,
        })
    return pd.DataFrame(rows, columns=["time_utc", "temp_c", "flag", "raw"]).sort_values("time_utc").reset_index(drop=True)


# ---------------------------------------------------------------- AMeDAS ----
def fetch_amedas(station: str, hours: int = 48, ttl: int = 55) -> pd.DataFrame:
    """Fichiers JMA de 3 h (pas de 10 min), heure Japon."""
    now_jst = datetime.now(timezone.utc) + timedelta(hours=9)
    start = now_jst - timedelta(hours=hours)
    rows = []
    t = start.replace(minute=0, second=0, microsecond=0)
    t = t.replace(hour=(t.hour // 3) * 3)
    while t <= now_jst:
        url = f"https://www.jma.go.jp/bosai/amedas/data/point/{station}/{t:%Y%m%d}_{t.hour:02d}.json"
        try:
            d = _get(url, ttl if t > now_jst - timedelta(hours=3) else 3600)
            for ts, v in d.items():
                tp = v.get("temp")
                if not tp or tp[1] != 0:  # [valeur, code qualité]
                    continue
                utc = pd.Timestamp(datetime.strptime(ts, "%Y%m%d%H%M%S"), tz="Asia/Tokyo").tz_convert("UTC")
                rows.append({"time_utc": utc, "temp_c": float(tp[0]), "flag": "", "raw": ""})
        except Exception:
            pass
        t += timedelta(hours=3)
    df = pd.DataFrame(rows, columns=["time_utc", "temp_c", "flag", "raw"])
    return df[df["time_utc"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)].sort_values("time_utc").reset_index(drop=True)


# -------------------------------------------------------------- Synoptic ----
def fetch_synoptic(icao: str, hours: int = 48, ttl: int = 240) -> pd.DataFrame:
    """ASOS US ~5 min via Synoptic Data (token gratuit)."""
    if not SYNOPTIC_TOKEN:
        return pd.DataFrame(columns=["time_utc", "temp_c", "flag", "raw"])
    url = (f"https://api.synopticdata.com/v2/stations/timeseries?stid={icao}"
           f"&recent={int(hours * 60)}&vars=air_temp&units=metric&token={SYNOPTIC_TOKEN}")
    d = _get(url, ttl)
    try:
        st0 = d["STATION"][0]
        times = st0["OBSERVATIONS"]["date_time"]
        temps = st0["OBSERVATIONS"]["air_temp_set_1"]
    except (KeyError, IndexError):
        return pd.DataFrame(columns=["time_utc", "temp_c", "flag", "raw"])
    rows = [{"time_utc": pd.Timestamp(t, tz="UTC"), "temp_c": float(v), "flag": "", "raw": ""}
            for t, v in zip(times, temps) if v is not None]
    return pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)


# ------------------------------------------------------------ Routage -------
def get_live_series(city_meta: dict, hours: int = 48):
    """Renvoie (df, info) — info = {source, cadence_min}. Meilleure source dispo."""
    icao = city_meta["icao"]
    if icao in AMEDAS_BY_ICAO:
        df = fetch_amedas(AMEDAS_BY_ICAO[icao], hours)
        if len(df):
            return df, {"source": "AMeDAS (JMA)", "cadence_min": 10}
    if icao.startswith("K") and SYNOPTIC_TOKEN:
        df = fetch_synoptic(icao, hours)
        if len(df):
            return df, {"source": "Synoptic (ASOS ~5 min)", "cadence_min": 5}
    return fetch_metar(icao, hours), {"source": "METAR (NOAA)",
                                      "cadence_min": 60 if icao.startswith("K") else 30}


def latest(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None
    r = df.iloc[-1]
    age = (pd.Timestamp.now(tz="UTC") - r["time_utc"]).total_seconds() / 60.0
    return {"time_utc": r["time_utc"], "temp_c": float(r["temp_c"]),
            "age_min": age, "raw": r.get("raw", "")}


def running_max_since_midnight_local(df: pd.DataFrame, tz: str) -> float | None:
    if df.empty:
        return None
    loc = df["time_utc"].dt.tz_convert(tz)
    sel = df.loc[loc >= pd.Timestamp.now(tz=tz).normalize(), "temp_c"]
    return float(sel.max()) if len(sel) else None


# ----------------------------------------------------------- Pic prévu ------
def peak_forecast(lat: float, lon: float, tz: str, ttl: int = 1800) -> list[dict]:
    """Pour aujourd'hui et demain (heure locale) : heure du pic, valeur, plateau."""
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=temperature_2m&timezone={tz.replace('/', '%2F')}&forecast_days=2")
    d = _get(url, ttl)
    s = pd.Series(d["hourly"]["temperature_2m"],
                  index=pd.DatetimeIndex(pd.to_datetime(d["hourly"]["time"])).tz_localize(tz))
    out = []
    for day, g in s.groupby(s.index.date):
        if g.isna().all():
            continue
        peak_t, peak_v = g.idxmax(), float(g.max())
        plat = g[g >= peak_v - 0.3]
        out.append({"date": day, "peak_local": peak_t, "peak_c": peak_v,
                    "plateau": (plat.index.min(), plat.index.max())})
    return out
