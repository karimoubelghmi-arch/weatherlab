"""Journaux PERSISTANTS des trades — DEUX fichiers distincts.

1) data/journal_logiciel.csv — le journal « RÉEL » du logiciel : chaque ordre
   envoyé via WeatherLab (réel, dry-run, papier, auto-trader) avec TOUTES les
   informations du scan (proba modèle, edges, Kelly, liquidité, tokenId,
   conditionId, market_id, negRisk…). Avant chaque batch de trades, une LIGNE
   DE SÉPARATION écrite dans le fichier lui-même détaille la stratégie exacte
   au moment de l'envoi (« LES TRADES DU … ONT ÉTÉ EFFECTUÉS SELON LA
   STRATÉGIE SUIVANTE : BANKROLL=… ; … »), lisible telle quelle dans Excel.
   La photo JSON complète (calibration + placement + contexte) est stockée
   dans la colonne `stratégie` de cette même ligne de séparation.

2) data/journal_wallet.csv — l'historique PUBLIC du wallet (adresse 0x…),
   agrégé en UNE SEULE LIGNE PAR POSITION (par tokenId) : d'abord remplie avec
   l'achat (prix moyen pondéré, montant, parts — les rachats cumulent), puis
   complétée à chaque « mise à jour » avec les ventes / redeem et le P&L
   réalisé. Le fichier est RECONSTRUIT entièrement à chaque import : il est
   donc toujours cohérent avec l'état on-chain, sans déduplication à gérer.

L'ancien fichier unique data/trade_journal.csv est migré automatiquement au
premier accès (lignes wallet → positions agrégées ; autres lignes → journal
logiciel avec leur stratégie), puis renommé en trade_journal.legacy.csv.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .data import DATA_DIR

LOGICIEL_PATH = DATA_DIR / "journal_logiciel.csv"
WALLET_PATH = DATA_DIR / "journal_wallet.csv"
LEGACY_PATH = DATA_DIR / "trade_journal.csv"
JOURNAL_PATH = LOGICIEL_PATH  # rétro-compatibilité (anciens imports / captions)

# préfixe des lignes de séparation « stratégie » dans journal_logiciel.csv
STRAT_PREFIX = "LES TRADES DU"

# --------------------------------------------------------------------------- #
#  Journal LOGICIEL (riche) — colonnes proches de l'export du scan
# --------------------------------------------------------------------------- #
_LCOLS = ["ville", "date_cible", "contrat", "côté",
          "proba_modèle_%", "prix_marché_%", "prix_côté_%", "edge_côté_%",
          "edge_effectif_%", "prix_moy_exéc_%", "liq_utile_$",
          "liquidité", "kelly_%", "mise_kelly_$", "mise_$",
          "mode", "order_type", "ok", "status", "orderID", "tx", "erreur",
          "tokenId", "conditionId", "market_id", "negRisk",
          "slug", "horodatage_utc", "stratégie"]


def strategy_snapshot(calib: dict | None, trading: dict | None, contexte: dict | None,
                      appris: dict | None = None) -> str:
    """Photo JSON de la stratégie complète, stockée sur la ligne de séparation.

    `appris` (V11.1) : paramètres CALIBRÉS par ville au moment du scan
    ({ville: {a, b, μ, σ, loi, …}}) — indispensables pour rejouer le batch,
    car quand use_ab=false les a/b de la barre latérale sont des champs
    manuels inutilisés (0), PAS les valeurs apprises."""
    snap = {"calibration": calib or {}, "placement": trading or {},
            "contexte": contexte or {}}
    if appris:
        snap["paramètres_appris"] = appris
    return json.dumps(snap, ensure_ascii=False)


def learned_from_infos(infos, villes=None) -> dict:
    """Extrait du DataFrame `infos` du scan les paramètres appris par ville.

    `villes` : restreint aux villes réellement tradées (sinon toutes).
    Tolérant aux colonnes absentes (anciens scans)."""
    out = {}
    try:
        if infos is None or not len(infos):
            return out
        want = None if villes is None else {str(v) for v in villes}
        cols = {"a": "a", "b": "b", "μ_°C": "μ", "σ_base_°C": "σ_base",
                "σ×": "σ×", "σ_°C": "σ", "spread": "spread", "loi": "loi",
                # V20 : photo complète — paramètres de loi, std(z), prévisions/modèle
                "loi_params": "loi_params", "std_z": "std_z",
                # V20.1 : « erreur d'hier » utilisée au scan (valeur en °C, ou
                # chaîne « — (raison) » si indisponible → repli neutre 0 dans μ)
                "err_hier_°C": "err_hier",
                "prévisions_modèles": "prévisions",
                "échéance_h": "échéance_h", "ancre_J2": "ancre_J2"}
        for _, r in infos.iterrows():
            v = str(r.get("ville"))
            if want is not None and v not in want:
                continue
            d = {}
            for src, dst in cols.items():
                val = r.get(src)
                if val is not None and val == val:
                    d[dst] = val if isinstance(val, str) else float(val)
            if d:
                out[v] = d
    except Exception:
        pass
    return out


def _get(row: dict, *keys, default=None):
    for k in keys:
        v = row.get(k)
        if v is not None and v == v:          # exclut NaN
            return v
    return default


def _fmt_num(v, suffix=""):
    try:
        f = float(v)
        s = f"{f:g}"
    except Exception:
        s = str(v)
    return s + suffix


def strategy_line_text(mode: str, snapshot: str | None, when: str | None = None) -> str:
    """Construit le texte humain de la ligne de séparation à partir de la photo JSON.

    Exemple : « LES TRADES DU 17/07/2026 À 14:32 UTC (MODE RÉEL) ONT ÉTÉ EFFECTUÉS
    SELON LA STRATÉGIE SUIVANTE : BANKROLL=1000$ ; FRACTION KELLY=25% ; … »
    """
    try:
        snap = json.loads(snapshot) if snapshot else {}
    except Exception:
        snap = {}
    pl = snap.get("placement") or {}
    cal = snap.get("calibration") or {}
    ctx = snap.get("contexte") or {}

    if when is None:
        when = datetime.now(timezone.utc).strftime("%d/%m/%Y À %H:%M UTC")
    head = f"{STRAT_PREFIX} {when} (MODE {str(mode).upper()}) ONT ÉTÉ EFFECTUÉS SELON LA STRATÉGIE SUIVANTE : "

    parts = []
    if pl:
        if pl.get("bankroll") is not None:
            parts.append(f"BANKROLL={_fmt_num(pl['bankroll'], '$')}")
        if pl.get("kelly_fraction") is not None:
            parts.append(f"FRACTION KELLY={_fmt_num(float(pl['kelly_fraction']) * 100, '%')}")
        if pl.get("min_edge") is not None:
            parts.append(f"EDGE MIN={_fmt_num(float(pl['min_edge']) * 100, ' pts')}")
        if pl.get("blend_w") is not None:
            # w est ESSENTIEL pour reproduire un batch au backtest : toujours journalisé.
            parts.append(f"POIDS MODÈLE w={_fmt_num(pl['blend_w'])}")
        if pl.get("min_trade") is not None:
            parts.append(f"MISE MIN={_fmt_num(pl['min_trade'], '$')}")
        if pl.get("side"):
            parts.append(f"CÔTÉ={pl['side']}")
        if pl.get("order_type"):
            parts.append(f"ORDRE={pl['order_type']}")
        ks = ctx.get("kill_switch") or {}
        if ks:
            parts.append(f"KILL-SWITCH={_fmt_num(ks.get('par_ordre_$'), '$')}/ordre, "
                         f"{_fmt_num(ks.get('par_24h_$'), '$')}/24h")
        elif pl.get("per_order_max_usd") is not None:
            parts.append(f"KILL-SWITCH={_fmt_num(pl['per_order_max_usd'], '$')}/ordre, "
                         f"{_fmt_num(pl.get('daily_max_usd'), '$')}/24h")
    # (la calibration complète reste dans la photo JSON, pas dans la ligne lisible)
    for k in ("scan", "heure_scan", "heure_envoi", "heure_scan_locale", "fenêtre", "date_cible"):
        if ctx.get(k):
            parts.append(f"{k.upper()}={ctx[k]}")
    if ctx.get("note"):
        parts.append(str(ctx["note"]).upper())

    if not parts:
        parts = ["STRATÉGIE EXACTE INCONNUE"]
    return head + " ; ".join(parts)


def log_trades(rows: list[dict], mode: str, snapshot: str,
               results: dict | None = None, signals: list[dict] | None = None,
               order_type: str = "") -> int:
    """Ajoute au journal LOGICIEL : une ligne de séparation « stratégie » puis une
    ligne par trade. `rows` = lignes du tableau d'opportunités du scan ;
    `results` (optionnel) = sortie de l'exécuteur indexée par id de signal ;
    `signals` permet la jointure ligne ↔ résultat. Renvoie le nb de trades."""
    if not rows:
        return 0
    sig_by_idx = list(signals or [])
    now = datetime.now(timezone.utc)
    now_s = now.strftime("%Y-%m-%d %H:%M:%S")

    # 1) ligne de séparation : texte humain dans la 1re colonne + JSON complet
    sep = {c: None for c in _LCOLS}
    sep["ville"] = strategy_line_text(mode, snapshot, when=now.strftime("%d/%m/%Y À %H:%M UTC"))
    sep["mode"] = mode
    sep["horodatage_utc"] = now_s
    sep["stratégie"] = snapshot

    out = [sep]
    for i, r in enumerate(rows):
        res = {}
        if results is not None and i < len(sig_by_idx):
            rr = results.get(sig_by_idx[i].get("id"), {})
            res = rr if isinstance(rr, dict) else {}
        out.append({
            "ville": _get(r, "ville"),
            "date_cible": _get(r, "date", "date_cible"),
            "contrat": _get(r, "contrat"), "côté": _get(r, "côté"),
            "proba_modèle_%": _get(r, "proba_modèle_%"),
            "prix_marché_%": _get(r, "prix_marché_%"),
            "prix_côté_%": _get(r, "prix_côté_%"),
            "edge_côté_%": _get(r, "edge_côté_%"),
            "edge_effectif_%": _get(r, "edge_effectif_%"),
            "prix_moy_exéc_%": _get(r, "prix_moy_exéc_%"),
            "liq_utile_$": _get(r, "liq_utile_$"),
            "liquidité": _get(r, "liquidité"),
            "kelly_%": _get(r, "kelly_%"),
            "mise_kelly_$": _get(r, "mise_kelly_$"),
            "mise_$": _get(r, "mise_$"),
            "mode": mode, "order_type": order_type,
            "ok": res.get("ok"), "status": res.get("status"),
            "orderID": res.get("orderID"), "tx": res.get("tx"),
            "erreur": res.get("error") or res.get("reason"),
            "tokenId": str(_get(r, "tokenId", default="")),
            "conditionId": str(_get(r, "conditionId", default="")),
            "market_id": str(_get(r, "market_id", default="")),
            "negRisk": _get(r, "negRisk"),
            "slug": _get(r, "slug", default=""),
            "horodatage_utc": now_s, "stratégie": None,
        })
    df = pd.DataFrame(out, columns=_LCOLS)
    header = not LOGICIEL_PATH.exists()
    df.to_csv(LOGICIEL_PATH, mode="a", header=header, index=False)
    return len(df) - 1


# V19 : DEUX statuts seulement — « DRY RUN » (aucun argent réellement misé :
# ex-papier, ex-dry-run, ex-auto-dry) et « RÉEL » (ordres envoyés avec de
# l'argent : ex-réel, ex-auto-réel). Les anciennes valeurs des fichiers
# existants sont normalisées à la lecture, le fichier n'est pas réécrit.
_MODE_MAP = {"papier": "DRY RUN", "dry-run": "DRY RUN", "dry_run": "DRY RUN",
             "auto-dry": "DRY RUN", "dry run": "DRY RUN", "dry": "DRY RUN",
             "réel": "RÉEL", "reel": "RÉEL", "auto-réel": "RÉEL", "auto-reel": "RÉEL"}


def normalize_mode(v) -> str:
    s = str(v).strip()
    low = s.lower()
    if low in _MODE_MAP:
        return _MODE_MAP[low]
    # motifs génériques : couvre aussi les anciens modes composés
    # (« auto-continu-dry », « réel (réintégré) »…)
    if "dry" in low or "papier" in low or "paper" in low:
        return "DRY RUN"
    if "réel" in low or "reel" in low or low == "real":
        return "RÉEL"
    return s.upper() if s and low != "nan" else s


def read_logiciel(with_strategy_rows: bool = False) -> pd.DataFrame:
    """Journal logiciel. Par défaut, les lignes de séparation « stratégie »
    sont retirées (analyse) ; with_strategy_rows=True les garde (affichage
    par batch : chaque groupe de trades suit sa ligne de stratégie)."""
    migrate_legacy()
    if not LOGICIEL_PATH.exists():
        return pd.DataFrame(columns=_LCOLS)
    try:
        df = pd.read_csv(LOGICIEL_PATH, dtype={"tokenId": str, "conditionId": str,
                                               "market_id": str})
    except Exception:
        return pd.DataFrame(columns=_LCOLS)
    if "mode" in df.columns:
        _m = df["ville"].astype(str).str.startswith(STRAT_PREFIX)
        df.loc[~_m, "mode"] = df.loc[~_m, "mode"].map(normalize_mode)
    if with_strategy_rows:
        return df
    mask = df["ville"].astype(str).str.startswith(STRAT_PREFIX)
    return df[~mask].reset_index(drop=True)


def read_journal() -> pd.DataFrame:
    """Rétro-compatibilité : renvoie les TRADES du journal logiciel (sans les
    lignes de séparation). Pour le wallet, utiliser read_wallet()."""
    return read_logiciel(with_strategy_rows=False)


def delete_logiciel_rows(indices: list[int]) -> int:
    """Supprime des lignes du journal PAPER TRADING (indices du fichier tel que
    lu par read_logiciel(with_strategy_rows=True)). Les lignes de stratégie
    devenues orphelines (plus aucun trade dans leur batch) sont retirées
    aussi. Renvoie le nombre de lignes restantes."""
    full = read_logiciel(with_strategy_rows=True)
    if not len(full):
        return 0
    full = full.drop(index=[i for i in indices if i in full.index]).reset_index(drop=True)
    is_sep = full["ville"].astype(str).str.startswith(STRAT_PREFIX)
    keep = []
    for i in range(len(full)):
        if not is_sep.iloc[i]:
            keep.append(i)
        elif i + 1 < len(full) and not is_sep.iloc[i + 1]:
            keep.append(i)                       # stratégie suivie d'au moins un trade
    full = full.iloc[keep]
    full.to_csv(LOGICIEL_PATH, index=False)
    return len(full)


# --------------------------------------------------------------------------- #
#  Journal WALLET — 1 ligne par position (tokenId), reconstruit à chaque MAJ
# --------------------------------------------------------------------------- #
_WCOLS = ["ville", "date_cible", "contrat", "côté", "statut",
          "nb_achats", "prix_achat_moy_%", "acheté_$", "parts",
          "nb_ventes", "prix_vente_moy_%", "vendu_$", "redeem_$",
          "encaissé_$", "pnl_réalisé_$",
          "premier_achat_utc", "dernier_mvt_utc", "slug", "tokenId"]


def read_wallet() -> pd.DataFrame:
    migrate_legacy()
    if not WALLET_PATH.exists():
        return pd.DataFrame(columns=_WCOLS)
    try:
        return pd.read_csv(WALLET_PATH, dtype={"tokenId": str})
    except Exception:
        return pd.DataFrame(columns=_WCOLS)


def _aggregate_wallet(movs: list[dict]) -> pd.DataFrame:
    """Agrège des mouvements normalisés en UNE ligne par tokenId.

    Mouvement : {token, ville, slug, date_cible, contrat, côté,
                 kind (BUY/SELL/REDEEM), usd, shares, price_pct, ts}.
    - Rachats : prix d'achat moyen PONDÉRÉ par les parts, montants cumulés.
    - Redeem : 1 part gagnante = 1 $ → parts rachetées = montant encaissé.
    - IMPORTANT : l'API Polymarket ne renseigne PAS le tokenId des REDEEM ;
      ils sont rattachés à leur position par (slug|ville+date_cible, contrat),
      en privilégiant la position qui a le plus de parts restantes.
    - P&L réalisé = (ventes + redeem) − achats (flux net si position ouverte).
    """
    def _tok(v):
        s = "" if v is None else str(v).strip()
        return "" if s.lower() in ("", "nan", "none") else s

    by_tok: dict[str, list[dict]] = {}
    orphans: list[dict] = []
    for m in movs:
        tok = _tok(m.get("token"))
        if tok:
            by_tok.setdefault(tok, []).append(m)
        else:
            orphans.append(m)                     # REDEEM sans tokenId (API)

    def _rem(ms):                                 # parts restantes d'un groupe
        b = sum(float(m.get("shares") or 0) for m in ms if m["kind"] == "BUY")
        s = sum(float(m.get("shares") or 0) for m in ms if m["kind"] == "SELL")
        r = sum(float(m.get("usd") or 0) for m in ms if m["kind"] == "REDEEM")
        return b - s - r

    def _key2(m):
        return (str(m.get("ville") or ""), str(m.get("date_cible") or ""),
                str(m.get("contrat") or ""))

    for o in orphans:
        cands = [tok for tok, ms in by_tok.items()
                 if (o.get("slug") and any(m.get("slug") == o.get("slug") and
                                           m.get("contrat") == o.get("contrat") for m in ms))
                 or any(_key2(m) == _key2(o) for m in ms)]
        if cands:
            best = max(cands, key=lambda t: _rem(by_tok[t]))
            by_tok[best].append(o)
        else:                                     # aucune position connue : ligne à part
            by_tok.setdefault(f"?{_key2(o)}", []).append(o)

    rows = []
    for tok, ms in by_tok.items():
        ms = sorted(ms, key=lambda x: x.get("ts") or "")
        buys = [m for m in ms if m["kind"] == "BUY"]
        sells = [m for m in ms if m["kind"] == "SELL"]
        reds = [m for m in ms if m["kind"] == "REDEEM"]

        def _tot(lst, k):
            return sum(float(m.get(k) or 0) for m in lst)

        b_usd, b_sh = _tot(buys, "usd"), _tot(buys, "shares")
        s_usd, s_sh = _tot(sells, "usd"), _tot(sells, "shares")
        r_usd = _tot(reds, "usd")
        p_buy = round(b_usd / b_sh * 100, 1) if b_sh > 0 else None
        p_sell = round(s_usd / s_sh * 100, 1) if s_sh > 0 else None
        remaining = b_sh - s_sh - r_usd            # redeem : 1 part = 1 $
        enc = round(s_usd + r_usd, 2)
        first = next((m for m in ms if m["kind"] == "BUY"), ms[0])

        if reds and r_usd > 0:
            statut = "✅ résolue (redeem)"
        elif reds:
            statut = "❌ résolue (perdue)"          # redeem à 0 $ = parts perdantes
        elif b_sh > 0 and s_sh > 0 and remaining <= max(1.0, 0.05 * b_sh):
            statut = "🔒 clôturée (vendue)"
        elif s_sh > 0 or s_usd > 0:
            statut = "🟠 partiellement vendue"
        else:
            statut = "🟢 ouverte"
            dc = first.get("date_cible")
            if dc:
                try:
                    from datetime import date, timedelta
                    if date.fromisoformat(str(dc)) < date.today() - timedelta(days=1):
                        statut = "🔚 expirée (rien encaissé — perdue ou redeem non réclamé)"
                except Exception:
                    pass

        rows.append({
            "ville": first.get("ville"), "date_cible": first.get("date_cible"),
            "contrat": first.get("contrat"), "côté": first.get("côté"),
            "statut": statut,
            "nb_achats": len(buys), "prix_achat_moy_%": p_buy,
            "acheté_$": round(b_usd, 2),
            "parts": round(b_sh, 2) if b_sh else None,
            "nb_ventes": len(sells), "prix_vente_moy_%": p_sell,
            "vendu_$": round(s_usd, 2), "redeem_$": round(r_usd, 2),
            "encaissé_$": enc, "pnl_réalisé_$": round(enc - b_usd, 2),
            "premier_achat_utc": first.get("ts"),
            "dernier_mvt_utc": ms[-1].get("ts"),
            "slug": first.get("slug", ""), "tokenId": tok,
        })
    df = pd.DataFrame(rows, columns=_WCOLS)
    if len(df):
        df = df.sort_values("dernier_mvt_utc", ascending=False).reset_index(drop=True)
    return df


_MOIS = ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"]


def _date_from_slug(slug: str):
    import re
    m = re.search(r"-on-([a-z]+)-(\d{1,2})-(\d{4})", slug or "")
    if not m or m.group(1) not in _MOIS:
        return None
    return f"{int(m.group(3)):04d}-{_MOIS.index(m.group(1)) + 1:02d}-{int(m.group(2)):02d}"


def import_wallet_trades(addr: str, progress=None) -> tuple[int, int]:
    """(Re)construit ENTIÈREMENT data/journal_wallet.csv depuis l'activité
    publique on-chain de `addr` (API data Polymarket, lecture seule) :
    UNE ligne par position (tokenId), achats/rachats moyennés, ventes et
    redeem intégrés. Relançable à volonté (bouton « mettre à jour ») : le
    fichier reflète toujours l'état réel, y compris les rachats intermédiaires.

    Renvoie (nb_positions, nb_mouvements_pris_en_compte).
    """
    from .portfolio import fetch_activity, city_from_slug, is_weather
    from .polymarket import parse_bucket

    acts = fetch_activity(addr.strip())
    weather = [a for a in acts if is_weather(a.get("slug", "")) and a.get("type") in ("TRADE", "REDEEM")]
    if progress:
        progress(0, max(1, len(weather)))

    movs = []
    for i, a in enumerate(sorted(weather, key=lambda x: x.get("timestamp", 0))):
        typ, side = a.get("type"), (a.get("side") or "").upper()
        kind = "REDEEM" if typ == "REDEEM" else side          # BUY / SELL / REDEEM
        if kind not in ("BUY", "SELL", "REDEEM"):
            continue
        ts = datetime.fromtimestamp(int(a.get("timestamp", 0)), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        usd = float(a.get("usdcSize", 0) or 0)
        shares = float(a.get("size", 0) or 0)
        if kind == "REDEEM" and shares <= 0:
            shares = usd                                       # 1 part gagnante = 1 $
        price = a.get("price")
        pb = parse_bucket(a.get("title", "") or "")
        movs.append({"token": str(a.get("asset", "")),
                     "ville": city_from_slug(a.get("slug", "")),
                     "slug": a.get("eventSlug") or a.get("slug", ""),
                     "date_cible": _date_from_slug(a.get("slug", "")),
                     "contrat": pb[2] if pb else (a.get("title") or "")[:60],
                     "côté": (a.get("outcome") or "").upper() or None,
                     "kind": kind, "usd": usd, "shares": shares,
                     "price_pct": float(price) * 100 if price not in (None, "") else None,
                     "ts": ts})
        if progress:
            progress(i + 1, len(weather))

    df = _aggregate_wallet(movs)
    df.to_csv(WALLET_PATH, index=False)
    return len(df), len(movs)


# --------------------------------------------------------------------------- #
#  Migration automatique de l'ancien data/trade_journal.csv (fichier unique)
# --------------------------------------------------------------------------- #
def migrate_legacy() -> None:
    """Si l'ancien trade_journal.csv existe : lignes « wallet » → positions
    agrégées dans journal_wallet.csv (si absent) ; autres lignes → journal
    logiciel, regroupées par batch (horodatage + stratégie) avec leur ligne
    de séparation. L'ancien fichier est renommé trade_journal.legacy.csv."""
    if not LEGACY_PATH.exists():
        return
    try:
        old = pd.read_csv(LEGACY_PATH, dtype={"tokenId": str})
    except Exception:
        LEGACY_PATH.rename(LEGACY_PATH.with_suffix(".legacy.csv"))
        return

    # --- 1) lignes wallet → positions agrégées -----------------------------
    w = old[old.get("mode") == "wallet"] if "mode" in old.columns else old.iloc[0:0]
    if len(w) and not WALLET_PATH.exists():
        movs = []
        for _, r in w.iterrows():
            ot = str(r.get("order_type") or "").upper()
            if ot not in ("BUY", "SELL", "REDEEM"):
                continue
            usd = float(r.get("mise_$") or 0)
            pc = r.get("prix_côté_%")
            pc = float(pc) if pc == pc and pc not in (None, "") else None
            shares = usd / (pc / 100) if pc and pc > 0 else (usd if ot == "REDEEM" else 0.0)
            movs.append({"token": str(r.get("tokenId") or ""),
                         "ville": r.get("ville"), "slug": r.get("slug", ""),
                         "date_cible": r.get("date_cible"), "contrat": r.get("contrat"),
                         "côté": r.get("côté") if r.get("côté") == r.get("côté") else None,
                         "kind": ot, "usd": usd, "shares": shares,
                         "price_pct": pc, "ts": r.get("horodatage_utc")})
        _aggregate_wallet(movs).to_csv(WALLET_PATH, index=False)

    # --- 2) autres lignes → journal logiciel, par batch --------------------
    nw = old[old.get("mode") != "wallet"] if "mode" in old.columns else old.iloc[0:0]
    if len(nw):
        out = []
        for (ts, snap), grp in nw.groupby(["horodatage_utc", "stratégie"], dropna=False):
            mode = str(grp["mode"].iloc[0])
            snap_s = snap if isinstance(snap, str) else None
            try:
                when = datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y À %H:%M UTC")
            except Exception:
                when = str(ts)
            sep = {c: None for c in _LCOLS}
            sep["ville"] = strategy_line_text(mode, snap_s, when=when)
            sep["mode"], sep["horodatage_utc"], sep["stratégie"] = mode, ts, snap_s
            out.append(sep)
            for _, r in grp.iterrows():
                d = {c: r.get(c) for c in _LCOLS if c in grp.columns}
                d["date_cible"] = r.get("date_cible")
                d["stratégie"] = None
                out.append({c: d.get(c) for c in _LCOLS})
        df = pd.DataFrame(out, columns=_LCOLS)
        df.to_csv(LOGICIEL_PATH, mode="a", header=not LOGICIEL_PATH.exists(), index=False)

    LEGACY_PATH.rename(LEGACY_PATH.with_name("trade_journal.legacy.csv"))
