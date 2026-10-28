"""Couche réseau robuste — reprise de la logique SSL de meteo_bot (certifi d'abord),
avec en dernier recours un contexte non vérifié pour les Mac dont les certificats
système ne sont pas installés (cause n°1 des « 0 villes » sur macOS).

Données publiques uniquement (météo, marchés) → le repli non vérifié est acceptable ici.
"""
from __future__ import annotations

import ssl
import time
import urllib.error
import urllib.request


def _make_ctxs():
    ctxs = []
    try:
        import certifi
        ctxs.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    try:
        ctxs.append(ssl.create_default_context())
    except Exception:
        pass
    try:
        ctxs.append(ssl._create_unverified_context())   # dernier recours (macOS sans certifs)
    except Exception:
        pass
    return ctxs


_CTXS = _make_ctxs()


def http_get(url: str, tries: int = 3, timeout: int = 40, headers: dict | None = None) -> str:
    h = {"User-Agent": "Mozilla/5.0"}
    if headers:
        h.update(headers)
    last = None
    for _ in range(tries):
        for ctx in _CTXS:
            try:
                req = urllib.request.Request(url, headers=h)
                return urllib.request.urlopen(req, timeout=timeout, context=ctx).read().decode()
            except urllib.error.URLError as e:
                last = e
            except Exception as e:
                last = e
        time.sleep(0.5)
    raise last
