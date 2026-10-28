"""Comparaison live avec Polymarket (optionnel, nécessite réseau).

Porté depuis meteo_bot.py mais branché sur le moteur WeatherLab 2 :
la proba modèle d'un bucket vient de la distribution prédictive (μ, σ, loi)
calculée par engine.run_pipeline. Tout échec réseau est géré proprement.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from .engine import cdf_T
from .net import http_get

INF = float("inf")
MOIS = ["january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december"]
SLUG_OVERRIDE: dict[str, str] = {}


def f2c(x):
    return (x - 32.0) / 1.8


def parse_bucket(question: str):
    """-> (lo_c, hi_c, label, unit). Bornes continues en °C (±inf possible)."""
    q = question
    m = re.search(r"(-?\d+)\s*(?:-|–|to)\s*(-?\d+)\s*°?\s*([CF])", q, re.I)
    if m:
        a, b, u = int(m.group(1)), int(m.group(2)), m.group(3).upper()
        lo, hi = a - 0.5, b + 0.5
    else:
        m = re.search(r"(-?\d+)\s*°?\s*([CF])", q, re.I)
        if not m:
            return None
        n, u = int(m.group(1)), m.group(2).upper()
        ql = q.lower()
        # « X or below/lower » inclut X ; « less than X » l'exclut (borne à X−0,5).
        if "less than" in ql:
            lo, hi = -INF, n - 0.5
        elif "below" in ql or "lower" in ql:
            lo, hi = -INF, n + 0.5
        # « X or higher/above » inclut X ; « greater/more than X » l'exclut.
        elif "greater than" in ql or "more than" in ql:
            lo, hi = n + 0.5, INF
        elif "higher" in ql or "above" in ql:
            lo, hi = n - 0.5, INF
        else:
            lo, hi = n - 0.5, n + 0.5
    lo_c = -INF if lo == -INF else (f2c(lo) if u == "F" else lo)
    hi_c = INF if hi == INF else (f2c(hi) if u == "F" else hi)
    if lo == -INF:
        lab = f"<={int(hi - 0.5)}{u}"
    elif hi == INF:
        lab = f">={int(lo + 0.5)}{u}"
    elif hi - lo > 1.0 + 1e-9:
        lab = f"{int(lo + 0.5)}-{int(hi - 0.5)}{u}"
    else:
        lab = f"{int(lo + 0.5)}{u}"
    return lo_c, hi_c, lab, u


def yes_price(mk: dict) -> float:
    prices = json.loads(mk.get("outcomePrices", "[]"))
    try:
        outs = json.loads(mk.get("outcomes", "[]"))
        for i, o in enumerate(outs):
            if str(o).strip().lower() == "yes":
                return float(prices[i])
    except Exception:
        pass
    return float(prices[0])



def side_token_and_price(mk: dict, side: str):
    """Retourne (tokenId, prix du côté demandé YES/NO).

    On mappe par nom d'outcome plutôt que par position supposée. Pour un pari NO,
    on achète le token NO, donc le prix du côté est le prix NO.
    """
    outs = json.loads(mk.get("outcomes", "[]"))
    toks = json.loads(mk.get("clobTokenIds", "[]"))
    prices = json.loads(mk.get("outcomePrices", "[]"))
    target = "yes" if str(side).upper() == "YES" else "no"
    for i, o in enumerate(outs):
        if str(o).strip().lower() == target:
            tok = str(toks[i]) if i < len(toks) else ""
            pr = float(prices[i]) if i < len(prices) else None
            return tok, pr
    i = 0 if target == "yes" else (1 if len(toks) > 1 else 0)
    tok = str(toks[i]) if i < len(toks) else ""
    pr = float(prices[i]) if i < len(prices) else None
    return tok, pr

def prob_bucket(lo_c, hi_c, mu, sigma, dist) -> float:
    p_lo = 0.0 if lo_c == -INF else float(cdf_T(lo_c, mu, sigma, dist))
    p_hi = 1.0 if hi_c == INF else float(cdf_T(hi_c, mu, sigma, dist))
    return max(0.0, p_hi - p_lo)


def market_slug(slug_city: str, date) -> str:
    city = SLUG_OVERRIDE.get(slug_city, slug_city.replace("_", "-"))
    return f"highest-temperature-in-{city}-on-{MOIS[date.month - 1]}-{date.day}-{date.year}"


def compare(slug_city: str, date, mu: float, sigma: float, dist, edge_threshold: float = 0.0) -> pd.DataFrame:
    """Récupère le marché du jour et compare proba modèle vs prix marché (edge signé YES)."""
    slug = market_slug(slug_city, date)
    try:
        ev = json.loads(http_get("https://gamma-api.polymarket.com/events?slug=" + slug))
    except Exception as exc:
        return pd.DataFrame([{"statut": "réseau indisponible", "détail": str(exc)[:80], "slug": slug}])
    if not ev:
        return pd.DataFrame([{"statut": "aucun marché", "détail": "slug non trouvé", "slug": slug}])
    rows = []
    for mk in ev[0].get("markets", []):
        pb = parse_bucket(mk.get("question", ""))
        if not pb:
            continue
        lo_c, hi_c, lab, u = pb
        try:
            py = yes_price(mk)
            pr = prob_bucket(lo_c, hi_c, mu, sigma, dist)
        except Exception:
            continue
        edge = pr - py
        if abs(edge) < edge_threshold:
            continue
        side = "YES" if edge >= 0 else "NO"
        token, side_price = side_token_and_price(mk, side)
        rows.append({
            "contrat": lab, "proba_modèle_%": round(pr * 100, 1), "prix_marché_%": round(py * 100, 1),
            "edge_YES_%": round(edge * 100, 1), "côté": side,
            "prix_côté_%": round(float(side_price) * 100, 1) if side_price is not None else None,
            "tokenId": token,
            "conditionId": mk.get("conditionId", ""),
            "market_id": mk.get("id", ""),
            "negRisk": bool(mk.get("negRisk")),
            "question": mk.get("question", ""),
        })
    return pd.DataFrame(rows).sort_values("edge_YES_%", key=lambda s: s.abs(), ascending=False).reset_index(drop=True) if rows else pd.DataFrame([{"statut": "aucun bucket exploitable", "slug": slug}])
