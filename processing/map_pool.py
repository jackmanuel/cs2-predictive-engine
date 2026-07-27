"""Map metadata and ordered Active Duty map-pool eras."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class MapDefinition:
    name: str
    aliases: tuple[str, ...] = ()
    hltv_bo1_abbreviation: str | None = None
    image_filename: str | None = None


@dataclass(frozen=True)
class MapPoolEraDefinition:
    name: str
    effective_from: datetime | None
    maps: tuple[str, ...]


# Add a metadata record only when a genuinely new map appears. Consumers derive
# canonical names, HLTV abbreviations, and report artwork from this registry.
MAP_DEFINITIONS = (
    MapDefinition("Ancient", ("anc",), "anc", "de_ancient.png"),
    MapDefinition("Anubis", ("anb",), "anb", "de_anubis.png"),
    MapDefinition("Cache", ("cch",), "cch", "de_cache.png"),
    MapDefinition("Cobblestone", ("cbl",), "cbl"),
    MapDefinition("Dust2", ("d2", "dust 2", "dust ii"), "d2", "de_dust2.png"),
    MapDefinition("Inferno", ("inf",), "inf", "de_inferno.png"),
    MapDefinition("Mirage", ("mrg",), "mrg", "de_mirage.png"),
    MapDefinition("Nuke", ("nk",), "nuke", "de_nuke.png"),
    MapDefinition("Overpass", ("over", "ovp"), "ovp", "de_overpass.png"),
    MapDefinition("Train", ("trn",), "trn", "de_train.png"),
    MapDefinition("Vertigo", ("vtg",), "vtg", "de_vertigo.png"),
)


# Append one entry for each pool rotation. The final entry is always the pool
# used for current simulations; effective dates are historical fallbacks, not
# assumptions that every event switched immediately.
MAP_POOL_ERAS = (
    MapPoolEraDefinition(
        name="overpass_active_duty",
        effective_from=None,
        maps=("Mirage", "Ancient", "Dust2", "Nuke", "Inferno", "Anubis", "Overpass"),
    ),
    MapPoolEraDefinition(
        name="cache_active_duty",
        effective_from=datetime(2026, 7, 8, tzinfo=timezone.utc),
        maps=("Mirage", "Ancient", "Dust2", "Nuke", "Inferno", "Anubis", "Cache"),
    ),
)


def _validate_registry() -> None:
    names = [definition.name for definition in MAP_DEFINITIONS]
    if len(names) != len(set(names)):
        raise ValueError("Map definitions must have unique canonical names")

    known_maps = set(names)
    previous_start = None
    for index, era in enumerate(MAP_POOL_ERAS):
        if len(era.maps) != 7 or len(set(era.maps)) != 7:
            raise ValueError(f"Map-pool era '{era.name}' must contain seven unique maps")
        unknown_maps = set(era.maps) - known_maps
        if unknown_maps:
            raise ValueError(f"Map-pool era '{era.name}' has undefined maps: {sorted(unknown_maps)}")
        if index == 0 and era.effective_from is not None:
            raise ValueError("The first map-pool era must have no effective start")
        if index > 0:
            if era.effective_from is None:
                raise ValueError(f"Map-pool era '{era.name}' needs an effective start")
            if previous_start is not None and era.effective_from <= previous_start:
                raise ValueError("Map-pool eras must be ordered by effective start")
        previous_start = era.effective_from


_validate_registry()

ACTIVE_MAP_POOL = MAP_POOL_ERAS[-1].maps
PREVIOUS_MAP_POOL = MAP_POOL_ERAS[-2].maps
MAP_POOL_CHANGE_AT = MAP_POOL_ERAS[-1].effective_from
SUPPORTED_VETO_MAPS = tuple(dict.fromkeys(map_name for era in MAP_POOL_ERAS for map_name in era.maps))

_MAP_DEFINITIONS_BY_NAME = {definition.name: definition for definition in MAP_DEFINITIONS}
MAP_ALIASES: dict[str, str] = {}
for _definition in MAP_DEFINITIONS:
    _keys = {
        _definition.name.lower(),
        f"de_{_definition.name.lower()}",
        *(_alias.lower() for _alias in _definition.aliases),
    }
    for _key in _keys:
        MAP_ALIASES[_key] = _definition.name

BO1_MAP_ABBREVIATIONS = frozenset(
    definition.hltv_bo1_abbreviation
    for definition in MAP_DEFINITIONS
    if definition.hltv_bo1_abbreviation
)


def canonical_map_name(raw_map: object) -> str | None:
    """Returns the canonical name for a registered map or alias."""
    key = str(raw_map or "").strip().lower()
    return MAP_ALIASES.get(key)


def map_image_filename(raw_map: object) -> str | None:
    """Returns the configured report image filename for a registered map."""
    canonical = canonical_map_name(raw_map)
    if canonical is None:
        return None
    return _MAP_DEFINITIONS_BY_NAME[canonical].image_filename


def map_pool_eras_with_bounds() -> tuple[tuple[MapPoolEraDefinition, datetime | None], ...]:
    """Returns each era paired with the next era's effective date as its end."""
    return tuple(
        (era, MAP_POOL_ERAS[index + 1].effective_from if index + 1 < len(MAP_POOL_ERAS) else None)
        for index, era in enumerate(MAP_POOL_ERAS)
    )
