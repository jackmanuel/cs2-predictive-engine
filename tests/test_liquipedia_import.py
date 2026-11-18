import json
from http import HTTPStatus
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

import dashboard_server
from ingestion import hltv_client


SINGAPORE_URL = "https://liquipedia.net/counterstrike/PGL/2026/Singapore"
SINGAPORE_STAGE_URLS = [f"{SINGAPORE_URL}/Stage_{stage}" for stage in range(1, 4)]


@pytest.fixture
def saved_links_path(monkeypatch, tmp_path):
    path = tmp_path / "data" / "liquipedia_links.json"
    monkeypatch.setattr(dashboard_server, "LIQUIPEDIA_LINKS_PATH", path)
    return path


@pytest.fixture
def event_alias_file(monkeypatch, tmp_path):
    alias_file = tmp_path / "liquipedia_team_aliases.csv"
    alias_file.write_text(
        "event_path,alias,team\n"
        "/counterstrike/PGL/2026/Singapore,GOSH,BC.Game\n"
        "/counterstrike/PGL/2026/Singapore,PVISION,PARIVISION\n"
        "/counterstrike/PGL/2026/Singapore,DENDELE CS,DENDELE\n"
        "/counterstrike/PGL/2026/Singapore,The One,1win\n"
        "/counterstrike/PGL/2026/Singapore,BB,BETBOOM\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_server, "LIQUIPEDIA_ALIASES_PATH", alias_file)


def matchlist(title, matches):
    rows = []
    for left, right in matches:
        rows.append(
            '<div class="brkts-matchlist-match">'
            f'<div class="brkts-matchlist-opponent"><span class="team-template-text">{left}</span></div>'
            '<div class="brkts-matchlist-score"></div>'
            f'<div class="brkts-matchlist-opponent"><span class="team-template-text">{right}</span></div>'
            '<div class="brkts-popup">Match details</div>'
            '</div>'
        )
    return (
        '<div class="general-collapsible brkts-matchlist">'
        f'<div class="brkts-matchlist-title"><b>{title}</b><button>Hide</button></div>'
        '<div class="brkts-matchlist-collapse-area">'
        '<div class="brkts-matchlist-date">November 24, 2026</div>'
        f'{"".join(rows)}</div></div>'
    )


def test_import_uses_round_one_match_rows_in_simulator_seed_order(
    monkeypatch, tmp_path, event_alias_file, saved_links_path
):
    matches = [
        ("HEROIC", "Imperial"),
        ("9z", "LVG"),
        ("NIP", "M80"),
        ("100T", "CW"),
        ('<a title="PVISION">PV</a>', "DENDELE CS"),
        ("Liquid", "GOSH"),
        ("FaZe", "Wildcard"),
        ("EF", "FLY"),
    ]
    html = matchlist("Round 2 Matches", [("Other A", "Other B")] * 8)
    html += matchlist("Round 1 Matches", matches) + '<table class="swisstable"></table>'
    requested_urls = []
    driver = SimpleNamespace(get=requested_urls.append, page_source=html)
    client = SimpleNamespace(driver=driver, start=lambda: None, stop=lambda: None)
    monkeypatch.setattr(hltv_client, "HLTVClient", lambda: client)
    monkeypatch.setattr(dashboard_server.time, "sleep", lambda _: None)
    monkeypatch.setattr(dashboard_server, "CLEAN_MAPS_PATH", tmp_path / "missing.parquet")

    url = f"{SINGAPORE_URL}/Stage_1"
    payload, status = dashboard_server.liquipedia_event_teams_payload({"url": url})

    assert requested_urls == [url]
    assert status == HTTPStatus.OK
    assert payload["saved_links"] == SINGAPORE_STAGE_URLS
    assert dashboard_server.load_liquipedia_links() == SINGAPORE_STAGE_URLS
    teams = payload["teams"]
    assert list(zip(teams[:8], teams[8:])) == [
        ("HEROIC", "Imperial"),
        ("9z", "Lynn Vision"),
        ("Ninjas in Pyjamas", "M80"),
        ("100 Thieves", "Chinggis Warriors"),
        ("PARIVISION", "DENDELE"),
        ("Liquid", "BC.Game"),
        ("FaZe", "Wildcard"),
        ("Eternal Fire", "FlyQuest"),
    ]


def test_saved_links_remember_stages_without_duplicates(saved_links_path):
    assert dashboard_server.load_liquipedia_links() == []
    for url in [*SINGAPORE_STAGE_URLS, SINGAPORE_STAGE_URLS[0]]:
        dashboard_server.save_liquipedia_link(url)

    assert dashboard_server.load_liquipedia_links() == SINGAPORE_STAGE_URLS


def test_removal_keeps_other_shortcuts_from_older_caches(saved_links_path):
    stage_one, stage_two, stage_three = SINGAPORE_STAGE_URLS
    other_event = "https://liquipedia.net/counterstrike/An_Event"
    saved_links_path.parent.mkdir()
    saved_links_path.write_text(json.dumps([stage_one, other_event]), encoding="utf-8")

    assert dashboard_server.load_liquipedia_links() == [*SINGAPORE_STAGE_URLS, other_event]
    remaining_links = [stage_one, stage_three, other_event]
    assert dashboard_server.remove_liquipedia_link(stage_two) == remaining_links
    assert dashboard_server.load_liquipedia_links() == remaining_links
    assert dashboard_server.remove_liquipedia_link(stage_two) == remaining_links
    assert dashboard_server.save_liquipedia_link(stage_three) == remaining_links

    assert dashboard_server.save_liquipedia_link(stage_two) == [*SINGAPORE_STAGE_URLS, other_event]


def test_invalid_links_are_not_saved(saved_links_path):
    payload, status = dashboard_server.liquipedia_event_teams_payload({
        "url": "https://example.com/liquipedia.net/counterstrike/PGL/2026/Singapore/Stage_1",
    })

    assert status == HTTPStatus.BAD_REQUEST
    assert "Invalid Liquipedia event URL" in payload["error"]
    assert not saved_links_path.exists()


@pytest.mark.parametrize(
    ("stage", "invites", "expected"),
    [
        (
            2,
            ["FUT", "Nemiga", "G2", "The One", "NAVI", "Astralis", "MongolZ", "MIBR"],
            ["FUT", "Nemiga", "G2", "1win", "Natus Vincere", "Astralis", "The MongolZ", "MIBR"],
        ),
        (
            3,
            ["MOUZ", "Vitality", "FURIA", "Aurora", "Spirit", '<a title="BB Team">BB</a>', "Falcons", "Legacy"],
            ["MOUZ", "Vitality", "FURIA", "Aurora", "Spirit", "BetBoom", "Falcons", "Legacy"],
        ),
    ],
)
def test_import_resolves_later_stage_invites(stage, invites, expected, event_alias_file):
    qualifiers = [f"Qualifier {seed}" for seed in range(1, 9)]
    soup = BeautifulSoup(matchlist("Round 1 Matches", zip(invites, qualifiers)), "html.parser")
    aliases = dashboard_server._load_liquipedia_event_aliases(f"{SINGAPORE_URL}/Stage_{stage}")

    assert dashboard_server._liquipedia_round_one_seeds(soup, {"BetBoom"}, aliases) == expected + qualifiers


def test_local_aliases_only_apply_to_the_configured_event(event_alias_file):
    assert dashboard_server._load_liquipedia_event_aliases(f"{SINGAPORE_URL}_Qualifiers/Stage_1") == {}


@pytest.mark.parametrize(
    ("matches", "error"),
    [
        ([("Alpha", "Bravo")] * 7, "eight two-team matches"),
        ([("Alpha", "Bravo")] * 7 + [("Alpha", '<a title="To Be Determined">TBD</a>')], "matchups are incomplete"),
        ([("Alpha", "Bravo")] * 8, "16 distinct teams"),
    ],
)
def test_import_rejects_incomplete_round_one_matches(matches, error):
    soup = BeautifulSoup(matchlist("Round 1 Matches", matches), "html.parser")

    with pytest.raises(ValueError, match=error):
        dashboard_server._liquipedia_round_one_seeds(soup, set())


def test_import_supports_legacy_swiss_standings():
    teams = [f"Team {seed}" for seed in range(1, 17)]
    rows = []
    for seed, name in enumerate(teams):
        opponent = teams[(seed + 8) % 16]
        rows.append(
            f'<tr><td>{seed + 1}</td><td>{name}</td><td></td><td></td><td></td><td></td>'
            f'<td><span class="team-template-team-icon"><a title="{opponent}"></a></span></td></tr>'
        )
    soup = BeautifulSoup(f'<table class="swisstable"><tr><th>Seed</th></tr>{"".join(rows)}</table>', "html.parser")

    expected_names = ["Team 1 Gaming", *teams[1:]]
    assert dashboard_server._liquipedia_round_one_seeds(soup, {"Team 1 Gaming"}) == expected_names
