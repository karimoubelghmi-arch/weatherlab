"""Portefeuille Polymarket en LECTURE SEULE, à partir de l'adresse PUBLIQUE.

Aucune clé privée, aucun mot de passe, aucune signature : on lit uniquement des données
publiques on-chain (API data Polymarket + historique de prix CLOB). Impossible de trader d'ici.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import pandas as pd

from .net import http_get

DATA = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def is_weather(slug: str) -> bool:
    return (slug or "").startswith("highest-temperature")


def city_from_slug(slug: str):
    m = re.search(r"highest-temperature-in-(.+?)-on-", slug or "")
    return m.group(1).replace("-", " ").title() if m else None


# ----------------------------- fetch -----------------------------
def fetch_value(addr: str):
    try:
        d = json.loads(http_get(f"{DATA}/value?user={addr}", timeout=30))
        return float(d[0]["value"]) if d else 0.0
    except Exception:
        return None


def fetch_positions(addr: str) -> list:
    return json.loads(http_get(f"{DATA}/positions?user={addr}&limit=500", timeout=45))


def fetch_activity(addr: str, max_rows: int = 8000) -> list:
    out, off, lim = [], 0, 500
    while off < max_rows:
        chunk = json.loads(http_get(f"{DATA}/activity?user={addr}&limit={lim}&offset={off}", timeout=45))
        if not chunk:
            break
        out += chunk
        if len(chunk) < lim:
            break
        off += lim
    return out


def fetch_price_history(asset: str, interval: str = "max", fidelity: int = 60) -> list:
    try:
        h = json.loads(http_get(f"{CLOB}/prices-history?market={asset}&interval={interval}&fidelity={fidelity}", timeout=30))
        return h.get("history", [])
    except Exception:
        return []


def fetch_pnl_series(addr: str, interval: str = "max", fidelity: str = "1d") -> pd.DataFrame:
    """Série P&L OFFICIELLE Polymarket (réalisé + non réalisé, tout le compte).
    La dernière valeur = le P&L courant affiché sur le site."""
    try:
        s = json.loads(http_get(
            f"https://user-pnl-api.polymarket.com/user-pnl?user_address={addr}&interval={interval}&fidelity={fidelity}",
            timeout=30))
        if not s:
            return pd.DataFrame()
        df = pd.DataFrame(s)
        df["datetime"] = pd.to_datetime(df["t"], unit="s")
        df["pnl_$"] = df["p"].round(2)
        return df[["datetime", "pnl_$"]]
    except Exception:
        return pd.DataFrame()


# ----------------------------- tables -----------------------------
def activity_df(acts: list) -> pd.DataFrame:
    rows = []
    for a in acts:
        slug = a.get("slug", "")
        typ, side = a.get("type"), a.get("side")
        usd = float(a.get("usdcSize", 0) or 0)
        if typ == "TRADE":
            cf = usd if side == "SELL" else -usd          # vendu = + ; acheté = -
        elif typ in ("REDEEM", "REWARD"):
            cf = usd                                       # gain encaissé
        else:
            cf = 0.0
        rows.append({
            "datetime": datetime.fromtimestamp(a["timestamp"]),
            "conditionId": a.get("conditionId"),
            "type": typ, "côté": side, "résultat": a.get("outcome"),
            "taille": a.get("size"), "prix": a.get("price"), "usdc": round(usd, 2),
            "cashflow": cf, "ville": city_from_slug(slug),
            "marché": a.get("title"), "slug": slug, "météo": is_weather(slug),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def positions_df(pos: list) -> pd.DataFrame:
    rows = []
    for p in pos:
        slug = p.get("slug", "")
        cur = p.get("curPrice", 0)
        resolved = bool(p.get("redeemable")) or cur in (0, 1)
        rows.append({
            "ville": city_from_slug(slug), "marché": p.get("title"),
            "résultat": p.get("outcome"), "statut": "résolue" if resolved else "ouverte",
            "taille": round(p.get("size", 0), 2), "prix_moy": round(p.get("avgPrice", 0), 3),
            "prix_actuel": round(cur, 3), "valeur_actuelle": round(p.get("currentValue", 0), 2),
            "P&L_$": round(p.get("cashPnl", 0), 2), "P&L_%": round(p.get("percentPnl", 0), 1),
            "fin": p.get("endDate"), "asset": p.get("asset"), "conditionId": p.get("conditionId"),
            "slug": slug, "météo": is_weather(slug),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ----------------------------- courbes -----------------------------
def realized_by_market(adf: pd.DataFrame, pdf: pd.DataFrame) -> pd.DataFrame:
    """P&L réalisé par marché CLOS = flux net par conditionId (fiable, redeem inclus),
    en excluant les marchés encore ouverts (dont le flux net = coût pas encore récupéré)."""
    if adf.empty:
        return pd.DataFrame()
    open_ids = set(pdf.loc[pdf["statut"] == "ouverte", "conditionId"]) if not pdf.empty else set()
    g = (adf.groupby("conditionId")
            .agg(pnl=("cashflow", "sum"), ville=("ville", "first"),
                 marché=("marché", "first"), météo=("météo", "first"),
                 dernier=("datetime", "max"))
            .reset_index())
    g["pnl"] = g["pnl"].round(2)
    g["statut"] = g["conditionId"].apply(lambda c: "ouvert" if c in open_ids else "clos")
    return g


def pnl_curve(adf: pd.DataFrame) -> pd.DataFrame:
    """Trésorerie nette cumulée dans le temps (achats −, ventes/gains +)."""
    d = adf.sort_values("datetime").copy()
    d["trésorerie_nette_$"] = d["cashflow"].cumsum().round(2)
    return d[["datetime", "trésorerie_nette_$", "type", "ville", "marché"]]


def open_value_curve(open_positions: list, progress=None, resample: str = "6h") -> pd.Series:
    """Valeur totale des positions ouvertes dans le temps (taille × prix historique du token)."""
    series = []
    n = len(open_positions)
    for i, p in enumerate(open_positions):
        if progress:
            progress(i, n)
        h = fetch_price_history(p["asset"])
        if not h:
            continue
        s = pd.Series({pd.Timestamp.fromtimestamp(pt["t"]): p.get("size", 0) * pt["p"] for pt in h})
        s = s[~s.index.duplicated(keep="last")]
        series.append(s)
    if not series:
        return pd.Series(dtype=float)
    wide = pd.concat(series, axis=1).sort_index().resample(resample).last().ffill()
    return wide.sum(axis=1)


def summary(adf: pd.DataFrame, pdf: pd.DataFrame, value: float) -> dict:
    """Chiffres FIABLES : P&L et taux de réussite calculés depuis l'activity (pas les positions,
    biaisées car les gagnantes encaissées disparaissent de la liste)."""
    trades = adf[adf["type"] == "TRADE"] if not adf.empty else adf
    rbm = realized_by_market(adf, pdf)
    closed = rbm[rbm["statut"] == "clos"] if not rbm.empty else rbm
    n_closed = len(closed)
    wins = int((closed["pnl"] > 0).sum()) if n_closed else 0
    open_p = pdf[pdf["statut"] == "ouverte"] if not pdf.empty else pdf
    return {
        "valeur_portefeuille_$": round(value, 2) if value is not None else None,
        "P&L_réalisé_marchés_clos_$": round(closed["pnl"].sum(), 2) if n_closed else 0.0,
        "taux_réussite_clos_%": round(100 * wins / n_closed, 1) if n_closed else None,
        "marchés_clos": n_closed,
        "nb_trades": int(len(trades)),
        "volume_tradé_$": round(trades["usdc"].sum(), 2) if not trades.empty else 0.0,
        "positions_ouvertes": int(len(open_p)),
        "valeur_positions_ouvertes_$": round(open_p["valeur_actuelle"].sum(), 2) if not open_p.empty else 0.0,
    }
