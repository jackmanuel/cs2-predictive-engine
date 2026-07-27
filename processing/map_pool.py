"""Canonical map names and Active Duty map-pool eras."""

from __future__ import annotations

from datetime import datetime, timezone


# Valve replaced Overpass with Cache when Premier Season Five began.
MAP_POOL_CHANGE_AT = datetime(2026, 7, 8, tzinfo=timezone.utc)

PREVIOUS_MAP_POOL = (
    "Mirage",
    "Ancient",
    "Dust2",
    "Nuke",
    "Inferno",
    "Anubis",
    "Overpass",
)

ACTIVE_MAP_POOL = (
    "Mirage",
    "Ancient",
    "Dust2",
    "Nuke",
    "Inferno",
    "Anubis",
    "Cache",
)

# Keep maps from the preceding era parseable so historical vetoes remain usable.
SUPPORTED_VETO_MAPS = tuple(dict.fromkeys((*PREVIOUS_MAP_POOL, *ACTIVE_MAP_POOL)))

MAP_ALIASES = {
    "anc": "Ancient",
    "ancient": "Ancient",
    "anb": "Anubis",
    "anubis": "Anubis",
    "cache": "Cache",
    "cch": "Cache",
    "d2": "Dust2",
    "de_ancient": "Ancient",
    "de_anubis": "Anubis",
    "de_cache": "Cache",
    "de_dust2": "Dust2",
    "de_inferno": "Inferno",
    "de_mirage": "Mirage",
    "de_nuke": "Nuke",
    "de_overpass": "Overpass",
    "dust2": "Dust2",
    "dust 2": "Dust2",
    "dust ii": "Dust2",
    "inf": "Inferno",
    "inferno": "Inferno",
    "mrg": "Mirage",
    "mirage": "Mirage",
    "nk": "Nuke",
    "nuke": "Nuke",
    "over": "Overpass",
    "overpass": "Overpass",
    "ovp": "Overpass",
}

# HLTV uses these values in place of "bo1" on some result rows.
BO1_MAP_ABBREVIATIONS = frozenset(
    {"mrg", "anc", "inf", "nuke", "anb", "d2", "cch", "ovp", "vtg", "trn", "cbl"}
)


def canonical_map_name(raw_map: object) -> str | None:
    """Returns the canonical name for an active or recently historical map."""
    key = str(raw_map or "").strip().lower()
    return MAP_ALIASES.get(key)
