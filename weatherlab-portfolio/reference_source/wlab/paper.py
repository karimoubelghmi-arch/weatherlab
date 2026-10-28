# -*- coding: utf-8 -*-
"""V11 — Résolution des marchés et PORTEFEUILLE PAPIER (journal_logiciel.csv).

Le journal PAPER TRADING note chaque trade avec son tokenId / conditionId.
Ce module ajoute la couche « et alors, ça a donné quoi ? » :

  refresh(trades)  → interroge Polymarket (gamma + CLOB, données PUBLIQUES,
                     aucune clé) pour chaque marché du journal :
                       - marché résolu ?  quel côté a payé ?
                       - marché encore ouvert ? prix actuel (midpoint CLOB)
                     et écrit le tout dans data/paper_resolutions.json.

  enrich(trades)   → recroise le journal avec ce cache, SANS réseau, et
                     renvoie les colonnes supplémentaires par trade :
                       statut_marché, prix_actuel_%, parts, valeur_$,
                       pnl_$, rendement_%.

Conventions P&L (1 part gagnante paie 1 $ au règlement) :
  parts   = mise_$ / prix_payé          (prix_moy_exéc_% si dispo, sinon prix_côté_%)
  résolu  : pnl = parts × payout − mise_$   (payout ∈ {0, 1})
  ouvert  : valeur = parts × prix_actuel ; pnl latent = valeur − mise_$
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .data import DATA_DIR
from .net import http_get

CACHE_PATH = DATA_DIR / "paper_resolutions.json"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# statuts affichés
ST_WON = "✅ résolu — gagné"
ST_LOST = "❌ résolu — perdu"
ST_OPEN = "🟢 ouvert"
ST_UNKNOWN = "⚪ non vérifié"


# --------------------------------------------------------------------------- #
#  cache disque
# --------------------------------------------------------------------------- #
def load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def last_refresh_utc() -> str | None:
    c = load_cache()
    ts = [v.get("vérifié_utc") for v in c.values() if isinstance(v, dict) and v.get("vérifié_utc")]
    return max(ts) if ts else None


# --------------------------------------------------------------------------- #
#  sélection des trades « positions » dans le journal
# --------------------------------------------------------------------------- #
def positions(trades: pd.DataFrame) -> pd.DataFrame:
    """Lignes du journal qui représentent une vraie position papier :
    mise > 0 et tokenId présent. Les lignes de stratégie sont déjà exclues
    par read_logiciel()."""
    if not len(trades):
        return trades
    df = trades.copy()
    df["mise_$"] = pd.to_numeric(df["mise_$"], errors="coerce").fillna(0.0)
    keep = (df["mise_$"] > 0) & df["tokenId"].astype(str).str.len().gt(10)
    return df[keep]


def _paid_price_pct(row) -> float | None:
    """Prix payé (%) : prix moyen d'exécution si renseigné, sinon prix du côté."""
    for col in ("prix_moy_exéc_%", "prix_côté_%"):
        try:
            v = float(row.get(col))
            if v == v and v > 0:
                return v
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- #
#  refresh réseau : gamma (résolution) + CLOB (prix actuels)
# --------------------------------------------------------------------------- #
def _fetch_market(condition_id: str) -> dict | None:
    try:
        d = json.loads(http_get(f"{GAMMA}/markets?condition_ids={condition_id}", timeout=30))
        return d[0] if d else None
    except Exception:
        return None


def _fetch_market_by_token(token_id: str) -> dict | None:
    """Repli pour les vieux trades sans conditionId : gamma sait chercher par tokenId."""
    try:
        d = json.loads(http_get(f"{GAMMA}/markets?clob_token_ids={token_id}", timeout=30))
        return d[0] if d else None
    except Exception:
        return None


def _fetch_last_price(token_id: str) -> float | None:
    """Dernier prix connu du token via l'historique CLOB (survit à la purge gamma).
    Un marché réglé finit épinglé à ~0 (perdu) ou ~1 (gagné)."""
    try:
        h = json.loads(http_get(f"{CLOB}/prices-history?market={token_id}&interval=max&fidelity=720",
                                timeout=30)).get("history", [])
        return float(h[-1]["p"]) if h else None
    except Exception:
        return None


def _cid_of(row, alias: dict) -> str | None:
    """conditionId d'une ligne : colonne si présente, sinon alias tokenId→cid du cache."""
    c = str(row.get("conditionId") or "")
    if c.startswith("0x"):
        return c
    return alias.get(str(row.get("tokenId") or ""))


def _fetch_mid(token_id: str) -> float | None:
    try:
        d = json.loads(http_get(f"{CLOB}/midpoint?token_id={token_id}", timeout=20))
        return float(d.get("mid"))
    except Exception:
        return None


def _pin_verdict(p: float | None):
    """Prix épinglé -> payout. ≥0,99 = gagné (1) ; ≤0,01 = perdu (0) ; sinon None."""
    if p is None:
        return None
    if p >= 0.99:
        return 1.0
    if p <= 0.01:
        return 0.0
    return None


def _token_resolved(entry: dict, tok: str) -> bool:
    """Ce token précis a-t-il déjà un verdict dans le cache ?"""
    if entry.get("résolu"):
        return True
    return tok in (entry.get("payout_tok") or {})


def refresh(trades: pd.DataFrame, progress=None) -> tuple[int, int]:
    """Met à jour le cache pour tous les marchés du journal.

    Renvoie (nb_marchés_vérifiés, nb_résolus). `progress(i, n)` optionnel.
    Un marché/token déjà résolu dans le cache n'est PAS re-téléchargé
    (une résolution est définitive).

    Chaîne de vérité, du plus riche au plus robuste :
      1. gamma par conditionId (résolution officielle + prix)
      2. gamma par tokenId (vieux trades sans conditionId)
      3. midpoint CLOB (prix actuel d'un marché ouvert)
      4. historique de prix CLOB (dernier prix épinglé à ~0/~1 = verdict —
         seule source qui survit à la purge des vieux marchés chez gamma)
    """
    pos = positions(trades)
    cache = load_cache()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    alias = cache.get("_alias") or {}          # tokenId -> conditionId (vieux trades)

    # tokens détenus, groupés par clé de cache (conditionId ou "tok:<tokenId>")
    held_by_key: dict[str, set] = {}
    for _, r in pos.iterrows():
        tok = str(r.get("tokenId") or "")
        if not tok:
            continue
        cid = _cid_of(r, alias)
        held_by_key.setdefault(cid or ("tok:" + tok), set()).add(tok)

    todo = [k for k, toks in held_by_key.items()
            if not all(_token_resolved(cache.get(k) or cache.get("tok:" + t) or {}, t)
                       for t in toks)]
    n = len(todo)
    for i, key in enumerate(todo):
        if progress:
            progress(i, n)
        is_cid = key.startswith("0x")
        toks_held = held_by_key[key]
        mk = _fetch_market(key) if is_cid else _fetch_market_by_token(next(iter(toks_held)))
        real_cid = key if is_cid else (mk or {}).get("conditionId")
        store_key = real_cid or key            # repli : clé "tok:<tokenId>"
        if not is_cid and real_cid:
            for t in toks_held:
                alias[t] = real_cid

        entry = cache.get(store_key) or {}
        entry["vérifié_utc"] = now

        prices, gtoks, resolved = [], [], False
        if mk and mk.get("outcomePrices"):      # gamma répond ET porte des données
            try:
                prices = [float(x) for x in json.loads(mk.get("outcomePrices", "[]"))]
            except Exception:
                prices = []
            try:
                gtoks = [str(t) for t in json.loads(mk.get("clobTokenIds", "[]"))]
            except Exception:
                gtoks = []
            closed = bool(mk.get("closed"))
            uma = str(mk.get("umaResolutionStatus", "")).lower()
            # résolu = marché clos (ou statut UMA « resolved ») ET prix figés en 0/1
            resolved = (closed or uma == "resolved") and prices and max(prices) > 0.99
            entry.update({"question": mk.get("question", ""), "clob_token_ids": gtoks,
                          "outcome_prices": prices, "fermé": closed, "résolu": bool(resolved)})

        if not resolved:
            # marché ouvert selon gamma, gamma muet, ou marché purgé :
            # prix actuel par token, et verdict par prix épinglé si possible
            mids = entry.get("mid") or {}
            payout_tok = entry.get("payout_tok") or {}
            for tok in toks_held:
                if tok in payout_tok:
                    continue
                m = _fetch_mid(tok)
                if m is not None:
                    mids[tok] = m                       # carnet vivant -> vraiment ouvert
                    continue
                p = _fetch_last_price(tok)              # carnet mort -> dernier prix connu
                v = _pin_verdict(p)
                if v is not None:
                    payout_tok[tok] = v                 # épinglé à 0/1 : verdict définitif
                    mids.pop(tok, None)
                elif p is not None:
                    mids[tok] = p
                elif gtoks and prices and tok in gtoks:
                    mids[tok] = prices[gtoks.index(tok)]   # dernier repli : prix gamma
            entry["mid"] = mids
            if payout_tok:
                entry["payout_tok"] = payout_tok
        cache[store_key] = entry
    if progress:
        progress(n, n)
    cache["_alias"] = alias
    save_cache(cache)
    n_res = sum(1 for k, toks in held_by_key.items()
                if all(_token_resolved(cache.get(k) or cache.get("tok:" + t) or {}, t)
                       for t in toks))
    return len(held_by_key), n_res


# --------------------------------------------------------------------------- #
#  enrichissement HORS-LIGNE du journal
# --------------------------------------------------------------------------- #
def enrich(trades: pd.DataFrame) -> pd.DataFrame:
    """Renvoie `positions(trades)` + colonnes statut_marché / prix_actuel_% /
    parts / valeur_$ / pnl_$ / rendement_%. Aucun appel réseau : tout vient du
    cache écrit par refresh()."""
    pos = positions(trades)
    if not len(pos):
        return pos
    cache = load_cache()
    alias = cache.get("_alias") or {}
    out = pos.copy()
    stat, cur, shares, value, pnl, ret = [], [], [], [], [], []
    for _, row in out.iterrows():
        cid, tok = _cid_of(row, alias) or "", str(row.get("tokenId"))
        mise = float(row["mise_$"])
        p = _paid_price_pct(row)
        sh = (mise / (p / 100.0)) if p else None
        e = cache.get(cid) or cache.get("tok:" + tok) or {}
        if not e:
            stat.append(ST_UNKNOWN); cur.append(None)
            shares.append(round(sh, 2) if sh else None)
            value.append(None); pnl.append(None); ret.append(None)
            continue
        ptok = (e.get("payout_tok") or {}).get(tok)
        if ptok is None and e.get("payout_token") is not None:
            ptok = e.get("payout_token")            # rétro-compat anciens caches
        if e.get("résolu") or ptok is not None:
            toks = e.get("clob_token_ids") or []
            prices = e.get("outcome_prices") or []
            payout = ptok
            if payout is None:
                payout = prices[toks.index(tok)] if tok in toks and len(prices) == len(toks) else None
            if payout is None or sh is None:
                stat.append(ST_UNKNOWN); cur.append(None)
                shares.append(round(sh, 2) if sh else None)
                value.append(None); pnl.append(None); ret.append(None)
                continue
            won = payout > 0.5
            val = sh * float(payout)
            stat.append(ST_WON if won else ST_LOST)
            cur.append(round(float(payout) * 100, 1))
            shares.append(round(sh, 2)); value.append(round(val, 2))
            pnl.append(round(val - mise, 2))
            ret.append(round((val - mise) / mise * 100, 1))
        else:
            mid = (e.get("mid") or {}).get(tok)
            if mid is None:
                mid = e.get("mid_token")
            stat.append(ST_OPEN)
            cur.append(round(float(mid) * 100, 1) if mid is not None else None)
            shares.append(round(sh, 2) if sh else None)
            if mid is not None and sh is not None:
                val = sh * float(mid)
                value.append(round(val, 2)); pnl.append(round(val - mise, 2))
                ret.append(round((val - mise) / mise * 100, 1))
            else:
                value.append(None); pnl.append(None); ret.append(None)
    out["statut_marché"] = stat
    out["prix_actuel_%"] = cur
    out["parts"] = shares
    out["valeur_$"] = value
    out["pnl_$"] = pnl
    out["rendement_%"] = ret
    return out


def summary(enriched: pd.DataFrame) -> dict:
    """Métriques agrégées du portefeuille papier."""
    if not len(enriched):
        return {}
    spent = float(enriched["mise_$"].sum())
    is_res = enriched["statut_marché"].isin([ST_WON, ST_LOST])
    is_open = enriched["statut_marché"].eq(ST_OPEN)
    val = pd.to_numeric(enriched["valeur_$"], errors="coerce")
    pnl = pd.to_numeric(enriched["pnl_$"], errors="coerce")
    return {
        "dépensé_$": round(spent, 2),
        "valeur_$": round(float(val.fillna(0).sum()), 2),
        "pnl_total_$": round(float(pnl.fillna(0).sum()), 2),
        "pnl_réalisé_$": round(float(pnl[is_res].fillna(0).sum()), 2),
        "pnl_latent_$": round(float(pnl[is_open].fillna(0).sum()), 2),
        "dépensé_ouvert_$": round(float(enriched.loc[is_open, "mise_$"].sum()), 2),
        "n_trades": int(len(enriched)),
        "n_ouverts": int(is_open.sum()),
        "n_résolus": int(is_res.sum()),
        "n_gagnés": int(enriched["statut_marché"].eq(ST_WON).sum()),
        "n_non_vérifiés": int(enriched["statut_marché"].eq(ST_UNKNOWN).sum()),
    }


def pnl_curve(enriched: pd.DataFrame) -> pd.DataFrame:
    """Courbe de P&L cumulé des trades fournis (résolus ET ouverts avec P&L
    connu), ordonnée par date cible (≈ date de règlement du marché)."""
    df = enriched.copy()
    df["pnl_$"] = pd.to_numeric(df["pnl_$"], errors="coerce")
    df = df.dropna(subset=["pnl_$"])
    if not len(df):
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date_cible"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "horodatage_utc"])
    daily = df.groupby(df["date"].dt.date)["pnl_$"].sum()
    out = daily.cumsum().to_frame("P&L_cumulé_$")
    out.index.name = "date"
    return out.reset_index()
