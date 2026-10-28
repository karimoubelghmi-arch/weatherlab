"""Liquidité du carnet d'ordres CLOB Polymarket.

Problème traité : le prix affiché n'est valable que pour les premières parts.
Un ordre marché plus gros que la profondeur « remonte » le carnet et paie un prix
moyen bien plus haut (slippage), ce qui peut détruire l'edge, voire le rendre négatif.

Principe : pour acheter un côté dont le modèle donne une proba p, chaque niveau du
carnet à prix q a un edge p − q. On n'accepte de consommer que les niveaux où
q ≤ p − edge_min. La somme de ces niveaux = « liquidité utile » : la mise maximale
exécutable au marché SANS descendre sous l'edge minimal. La mise Kelly est écrêtée
dessus ; si même la mise minimale ne passe pas, le trade est bloqué.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .net import http_get

CLOB = "https://clob.polymarket.com"


def fetch_asks(token_id: str):
    """Côté vente du carnet pour un token (= ce qu'on consomme en ACHETANT ce token).

    Renvoie [[prix, taille], …] trié par prix croissant, ou None si indisponible.
    """
    if not token_id:
        return None
    try:
        b = json.loads(http_get(f"{CLOB}/book?token_id={token_id}", timeout=20))
        asks = sorted(([float(a["price"]), float(a["size"])] for a in (b.get("asks") or [])),
                      key=lambda x: x[0])
        return asks or None
    except Exception:
        return None


def fetch_books(token_ids, workers: int = 8, progress=None) -> dict:
    """Carnets de plusieurs tokens en parallèle. {token_id: asks | None}."""
    toks = [t for t in dict.fromkeys(str(t) for t in token_ids) if t]
    out = {}
    if not toks:
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_asks, t): t for t in toks}
        for i, f in enumerate(as_completed(futs)):
            out[futs[f]] = f.result()
            if progress:
                progress(i + 1, len(toks))
    return out


def usable_liquidity(asks, price_limit: float) -> float:
    """$ total disponible aux niveaux de prix ≤ price_limit (edge conservé niveau par niveau)."""
    return float(sum(p * s for p, s in asks if p <= price_limit + 1e-9))


def max_stake_for_avg_price(asks, price_limit: float) -> float:
    """Mise maximale ($) telle que le PRIX MOYEN d'exécution reste ≤ price_limit.

    Critère « edge effectif » : on juge la mise sur son prix moyen (donc son edge moyen),
    pas niveau par niveau. On peut donc consommer une partie des niveaux au-dessus de la
    limite tant que la moyenne pondérée reste sous la limite.

    Marche : on consomme les niveaux par prix croissant. Tant que q ≤ limite, tout est bon.
    Pour un niveau q > limite, la quantité x ($) encore absorbable sans dépasser la moyenne
    vérifie (S + x)/(N + x/q) ≤ L, soit x ≤ (L·N − S)/(1 − L/q) ; au-delà, la moyenne
    atteint exactement L et plus rien n'est absorbable.
    """
    L = float(price_limit)
    if L <= 0:
        return 0.0
    spent = shares = 0.0
    for p, s in asks:
        cap_lvl = p * s
        if p <= L + 1e-9:
            spent += cap_lvl
            shares += s
            continue
        room = (L * shares - spent) / (1.0 - L / p)
        if room > 1e-9:
            take = min(room, cap_lvl)
            spent += take
            shares += take / p
            if take < cap_lvl - 1e-9:
                break
        else:
            break
    return float(spent)


def exec_stats(asks, usd: float):
    """Simule un ordre marché de `usd` $ en remontant le carnet.

    Renvoie (parts obtenues, prix moyen payé, $ réellement dépensés).
    Si le carnet est épuisé avant `usd`, dépensé < usd.
    """
    spent = shares = 0.0
    for p, s in asks:
        take = min(usd - spent, p * s)
        if take <= 1e-12:
            break
        shares += take / p
        spent += take
    avg = (spent / shares) if shares > 0 else None
    return shares, avg, spent
