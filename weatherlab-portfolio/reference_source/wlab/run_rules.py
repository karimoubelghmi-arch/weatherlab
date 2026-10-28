# -*- coding: utf-8 -*-
"""Règle déterministe « quel run était en vigueur à l'instant T ? » par modèle.

C'est la pièce centrale de la cohérence entraînement = trading :
la MÊME fonction sert à reconstruire le socle (passé) et peut servir au live.

run_in_effect(model, decision_utc) → datetime d'init du dernier run dont on est
sûr qu'il était PUBLIÉ à decision_utc :  init + latence + marge ≤ decision.
La marge (30 min) absorbe l'oscillation de la latence ; en cas de doute on prend
le cycle précédent — prudent, et surtout identique des deux côtés.

Latences par défaut MESURÉES le 06/08/2026 via l'API de métadonnées Open-Meteo
(https://api.open-meteo.com/data/{id}/static/meta.json). Relançables via
measure_latencies() ; les valeurs mesurées sont persistées dans
data/run_latencies.json et priment sur les défauts.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LATENCY_FILE = DATA_DIR / "run_latencies.json"
MARGIN_H = 0.5

# Heures d'init UTC des cycles, par modèle (défaut : 4 cycles globaux)
CYCLES: dict[str, list[int]] = {
    "ecmwf_ifs025": [0, 6, 12, 18],
    "gfs_seamless": [0, 6, 12, 18],
    "icon_global": [0, 6, 12, 18],
    "jma_seamless": [0, 6, 12, 18],
    "ukmo_seamless": [0, 6, 12, 18],
    "gem_global": [0, 12],
    "cma_grapes_global": [0, 6, 12, 18],
    "icon_eu": [0, 3, 6, 9, 12, 15, 18, 21],
    "meteofrance_arpege_europe": [0, 6, 12, 18],
    "knmi_harmonie_arome_europe": list(range(24)),
    "dmi_harmonie_arome_europe": list(range(0, 24, 3)),
    "metno_nordic": list(range(24)),
    "gfs_hrrr": list(range(24)),
    "gem_hrdps_continental": [0, 6, 12, 18],
}

# Latence de PUBLICATION (h après l'init) — mesures du 06/08/2026 + défauts prudents
DEFAULT_LATENCY_H: dict[str, float] = {
    "ecmwf_ifs025": 8.0,
    "gfs_seamless": 5.7,
    "icon_global": 3.6,
    "jma_seamless": 9.8,
    "ukmo_seamless": 8.1,
    "gem_global": 7.0,
    "cma_grapes_global": 9.5,
    "icon_eu": 3.0,
    "meteofrance_arpege_europe": 5.0,
    "knmi_harmonie_arome_europe": 2.5,
    "dmi_harmonie_arome_europe": 2.5,
    "metno_nordic": 1.5,
    "gfs_hrrr": 1.5,
    "gem_hrdps_continental": 4.0,
}
_FALLBACK_LATENCY_H = 9.0  # modèle inconnu : très prudent

# id de l'API de métadonnées quand il diffère du nom Open-Meteo « forecast »
META_IDS: dict[str, list[str]] = {
    "gfs_seamless": ["ncep_gfs013"],
    "icon_global": ["dwd_icon"],
    "icon_eu": ["dwd_icon_eu"],
    "jma_seamless": ["jma_gsm"],
    "ukmo_seamless": ["ukmo_global_deterministic_10km"],
    "gem_global": ["cmc_gem_gdps"],
    "gem_hrdps_continental": ["cmc_gem_hrdps"],
    "gfs_hrrr": ["ncep_hrrr_conus"],
    "meteofrance_arpege_europe": ["meteofrance_arpege_europe025",
                                  "meteofrance_arpege_europe"],
    "knmi_harmonie_arome_europe": ["knmi_harmonie_arome_europe"],
    "dmi_harmonie_arome_europe": ["dmi_harmonie_arome_europe"],
    "metno_nordic": ["metno_nordic_pp", "metno_nordic"],
}


def _load_measured() -> dict:
    if LATENCY_FILE.exists():
        try:
            return json.loads(LATENCY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def latency_h(model: str) -> float:
    meas = _load_measured().get(model)
    if isinstance(meas, dict) and isinstance(meas.get("latency_h"), (int, float)):
        return float(meas["latency_h"])
    return DEFAULT_LATENCY_H.get(model, _FALLBACK_LATENCY_H)


def run_in_effect(model: str, decision_utc: datetime) -> datetime:
    """Init (UTC) du dernier run publié avec certitude à decision_utc."""
    if decision_utc.tzinfo is None:
        decision_utc = decision_utc.replace(tzinfo=timezone.utc)
    cyc = sorted(CYCLES.get(model, [0, 6, 12, 18]))
    lat = latency_h(model) + MARGIN_H
    limit = decision_utc - timedelta(hours=lat)
    d = limit.date()
    for _ in range(3):  # au pire on remonte 2 jours (modèles 2 cycles/jour)
        cand = [datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc)
                for h in cyc
                if datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc) <= limit]
        if cand:
            return max(cand)
        d = d - timedelta(days=1)
    raise RuntimeError(f"aucun run trouvé pour {model} avant {decision_utc}")


def prev_cycle(model: str, init_utc: datetime) -> datetime:
    """Le cycle juste avant init_utc (sans latence — pour replis d'archive)."""
    cyc = sorted(CYCLES.get(model, [0, 6, 12, 18]))
    d = init_utc.date()
    for _ in range(3):
        cand = [datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc)
                for h in cyc
                if datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc) < init_utc]
        if cand:
            return max(cand)
        d = d - timedelta(days=1)
    raise RuntimeError(f"aucun cycle avant {init_utc} pour {model}")


def age_at_h(model: str, decision_utc: datetime, at_utc: datetime) -> float:
    """Âge (h) de l'info du run en vigueur à decision_utc, mesuré à at_utc (ex. le pic)."""
    init = run_in_effect(model, decision_utc)
    return (at_utc - init).total_seconds() / 3600.0


def measure_latencies(models: list[str] | None = None, verbose: bool = True) -> dict:
    """Interroge l'API méta et persiste les latences mesurées dans data/run_latencies.json.
    Garde-fou : une méta gelée (init plus vieux que 3 cycles) est ignorée."""
    from .net import http_get
    out = _load_measured()
    now = datetime.now(timezone.utc)
    for m in (models or list(CYCLES)):
        ids = META_IDS.get(m, [m])
        for mid in ids:
            try:
                d = json.loads(http_get(
                    f"https://api.open-meteo.com/data/{mid}/static/meta.json"))
                init = d.get("last_run_initialisation_time")
                avail = d.get("last_run_availability_time")
                upd = float(d.get("update_interval_seconds") or 21600)
                if not init or not avail:
                    continue
                init_dt = datetime.fromtimestamp(init, timezone.utc)
                if (now - init_dt).total_seconds() > 3 * upd:
                    if verbose:
                        print(f"  ⚠️ {m} ({mid}) : méta gelée (init {init_dt:%d/%m %H:%M}), ignorée")
                    continue
                lat = (avail - init) / 3600.0
                out[m] = {"latency_h": round(lat, 2), "meta_id": mid,
                          "measured_utc": now.isoformat(timespec="seconds")}
                if verbose:
                    print(f"  ✓ {m:<28} latence {lat:5.2f} h (via {mid})")
                break
            except Exception:
                continue
        else:
            if verbose:
                print(f"  – {m:<28} non mesurable → défaut {latency_h(m)} h")
    LATENCY_FILE.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out
