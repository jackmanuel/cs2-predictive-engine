from http import HTTPStatus
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

import dashboard_server
from ingestion import hltv_client


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


def test_import_uses_round_one_match_rows_in_simulator_seed_order(monkeypatch, tmp_path):
    matches = [
        ("HEROIC", "Imperial"),
        ("9z", "LVG"),
        ("NIP", "M80"),
        ("100T", "CW"),
        ("PV", "DENDELE"),
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

    url = "https://liquipedia.net/counterstrike/PGL/2026/Singapore/Stage_1"
    payload, status = dashboard_server.liquipedia_event_teams_payload({"url": url})

    assert requested_urls == [url]
    assert status == HTTPStatus.OK
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


@pytest.mark.parametrize(
    ("matches", "error"),
    [
        ([("Alpha", "Bravo")] * 7, "eight two-team matches"),
        ([("Alpha", "Bravo")] * 7 + [("Alpha", "TBD")], "identify all 16 teams"),
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
