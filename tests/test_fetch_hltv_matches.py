import json
import logging

import pytest

from ingestion import fetch_hltv_matches as scraper
from ingestion.hltv_client import HLTVClient


def match(number, date="2026-10-01", team1="G2", team2="Other"):
    return {
        "url": f"https://www.hltv.org/matches/{number}/example",
        "date": date,
        "team1": team1,
        "team2": team2,
    }


class FakeClient:
    def __init__(self):
        self.pages = []
        self.scanned = []
        self.details = []
        self.stats = []
        self.failures = set()
        self.stopped = False

    def fetch_recent_results(self, *, pages, start_page):
        self.scanned.extend(range(start_page, start_page + pages))
        return [row for page in self.pages[start_page:start_page + pages] for row in page]

    def fetch_match_details(self, url):
        self.details.append(url)
        if url in self.failures:
            raise RuntimeError("test detail failure")
        return {"maps": [{"stats_url": f"{url}/stats"}]}

    def fetch_map_stats(self, url):
        self.stats.append(url)
        return {"round_history": "rounds"}

    def stop(self):
        self.stopped = True


@pytest.fixture
def run(tmp_path, monkeypatch, caplog):
    path = tmp_path / "raw" / "hltv_matches.json"
    client = FakeClient()
    monkeypatch.setattr(scraper, "HLTV_MATCHES_FILE", path)
    monkeypatch.setattr(scraper, "HLTVClient", lambda: client)
    caplog.set_level(logging.INFO)
    return client, path


@pytest.fixture
def whitelist(tmp_path):
    path = tmp_path / "teams.csv"
    path.write_text("team,stage\n  G2  ,1\nThe   MongolZ,2\nTHE mongolz,3\n ,1\n", encoding="utf-8-sig")
    return path


def test_load_whitelist(whitelist):
    assert scraper.load_team_whitelist(whitelist) == {"g2": "G2", "the mongolz": "The MongolZ"}


@pytest.mark.parametrize("content, message", [
    (None, "Cannot read"),
    ("name\nG2\n", "team"),
    ("team,stage\n ,1\n", "no usable teams"),
    ('team\n"unterminated\n', "Cannot read"),
    ("team,stage\nG2\n", "wrong number of columns"),
])
def test_invalid_whitelist(tmp_path, content, message):
    path = tmp_path / "teams.csv"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        scraper.load_team_whitelist(path)


@pytest.mark.parametrize("mode, numbers", [("any", [1, 2]), ("both", [2])])
def test_team_filter_before_details(run, whitelist, mode, numbers):
    client, path = run
    rows = [match(1, team1="g2"), match(2, team2=" the  MONGOLZ "), match(3, team1="G2 Academy")]
    client.pages = [rows]
    assert scraper.main(["--team-whitelist", str(whitelist), "--team-match", mode]) == 0
    expected = [rows[number - 1]["url"] for number in numbers]
    assert client.details == expected
    assert client.stats == [f"{url}/stats" for url in expected]
    assert [record["url"] for record in json.loads(path.read_text())] == expected
    assert client.stopped


@pytest.mark.parametrize("args", [
    ["--start-date", "2026-02-30"],
    ["--end-date", "20261001"],
    ["--start-date", "2026-10-02", "--end-date", "2026-10-01"],
    ["--pages", "0"],
    ["--matches", "-1"],
    ["--team-match", "any"],
])
def test_invalid_cli(run, monkeypatch, args):
    monkeypatch.setattr(scraper, "HLTVClient", lambda: pytest.fail("client initialized"))
    with pytest.raises(SystemExit) as error:
        scraper.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("args, numbers", [
    (["--start-date", "2026-09-03", "--end-date", "2026-11-02"], [4, 5]),
    (["--start-date", "2026-09-03"], [1, 2, 3, 4, 5]),
    (["--end-date", "2026-11-02"], [4, 5, 6, 7]),
])
def test_date_bounds_and_full_boundary_page(run, caplog, args, numbers):
    client, _ = run
    client.pages = [
        [match(1, "2026-11-16")],
        [match(2, "2026-11-15")],
        [match(3, "2026-11-03")],
        [match(4, "2026-11-02")],
        [match(5, "2026-09-03"), match(6, "2026-09-02")],
        [match(7, "2026-09-01")],
    ]
    assert scraper.main(args) == 0
    assert client.details == [match(number)["url"] for number in numbers]
    has_start = "--start-date" in args
    assert client.scanned == list(range(5 if has_start else 7))
    assert f"{6 if has_start else 7} unique" in caplog.text


def test_unknown_dates_keep_boundary_open(run, caplog):
    client, path = run
    client.pages = [
        [match(1, "2026-09-02"), match(2, None)],
        [match(3, "unparseable"), match(4, "2026-09-01")],
        [match(5, "2026-08-31")],
    ]
    assert scraper.main(["--start-date", "2026-09-03", "--dry-run"]) == 0
    assert client.scanned == [0, 1, 2]
    assert "Unknown dates = 2" in caplog.text
    assert "stopped by date boundary" in caplog.text
    assert not path.parent.exists()


def test_out_of_order_page_does_not_cross_boundary_early(run, caplog):
    client, _ = run
    client.pages = [[match(1, "2026-08-30"), match(2), match(3, "2026-08-29")], [match(4)]]
    assert scraper.main(["--start-date", "2026-09-03", "--dry-run"]) == 0
    assert client.scanned == [0, 1, 2]
    assert "Selected 2" in caplog.text


def test_repeated_page_is_abnormal(run, caplog):
    client, path = run
    client.pages = [[match(1), match(2)], [match(2), match(1)]]
    assert scraper.main(["--end-date", "2026-11-02"]) == 1
    assert client.scanned == [0, 1]
    assert not client.details and not path.exists()
    assert client.stopped
    assert "abnormal pagination" in caplog.text


@pytest.mark.parametrize("limit, selected, reason", [(1, 0, "page cap"), (5, 2, "match cap")])
def test_combined_limits_after_eligibility_and_dedup(run, whitelist, caplog, limit, selected, reason):
    client, _ = run
    client.pages = [
        [match(1, team1="Spirit")],
        [match(2), match(2)],
        [match(2), match(3), match(4)],
        [match(5)],
    ]
    assert scraper.main(["--pages", str(limit), "--matches", "2", "--team-whitelist", str(whitelist)]) == 0
    assert len(client.details) == selected
    assert client.scanned == list(range(1 if limit == 1 else 3))
    assert f"stopped by {reason}" in caplog.text


@pytest.mark.parametrize("use_whitelist", [False, True])
def test_undated_default_scans_three_pages(run, whitelist, use_whitelist):
    client, _ = run
    client.pages = [[match(number)] for number in range(4)]
    args = ["--team-whitelist", str(whitelist)] if use_whitelist else []
    assert scraper.main(args) == 0
    assert client.scanned == [0, 1, 2]
    assert len(client.details) == 3


def test_dry_run_leaves_canonical_file_unchanged(run, whitelist, caplog):
    client, path = run
    path.parent.mkdir()
    path.write_bytes(b"[]")
    client.pages = [[
        match(3, "2026-10-02"),
        match(1, team2="The MongolZ"),
        match(2, team1="Spirit"),
    ]]
    assert scraper.main([
        "--end-date", "2026-10-01", "--team-whitelist", str(whitelist), "--dry-run",
    ]) == 0
    assert client.scanned and not client.details and not client.stats
    assert path.read_bytes() == b"[]"
    assert "The MongolZ | 1 selected" in caplog.text
    assert "G2 | 1 selected" in caplog.text
    assert match(1)["url"] in caplog.text
    assert match(3)["url"] not in caplog.text
    assert "Outside date bounds = 1" in caplog.text
    assert "Team excluded = 1" in caplog.text


@pytest.mark.parametrize("retry", [False, True])
def test_resume_and_retry_in_place(run, retry):
    client, path = run
    success, failure, repeated_failure = match(1), match(2), match(3)
    client.pages = [[success, failure, repeated_failure, match(4)]]
    client.failures.add(repeated_failure["url"])
    records = [success, {**failure, "error": "old"}, {**repeated_failure, "error": "old"}]
    path.parent.mkdir()
    path.write_text(json.dumps(records))
    assert scraper.main(["--retry-errors"] if retry else []) == 0
    expected = [failure["url"], repeated_failure["url"]] if retry else []
    assert client.details == [*expected, match(4)["url"]]
    saved = json.loads(path.read_text())
    assert len(saved) == 4 and saved[0] == success
    assert ("error" in saved[1]) == (not retry)
    assert saved[2]["error"] == ("test detail failure" if retry else "old")


def test_third_consecutive_failure_is_saved(run):
    client, path = run
    client.pages = [[match(number) for number in range(4)]]
    client.failures = {row["url"] for row in client.pages[0]}
    assert scraper.main([]) == 1
    saved = json.loads(path.read_text())
    assert [row["url"] for row in saved] == client.details == [match(i)["url"] for i in range(3)]
    assert all("error" in row for row in saved)
    assert client.stopped


def test_failed_atomic_replace_preserves_canonical_data(run, monkeypatch):
    client, path = run
    client.pages = [[match(1)]]
    path.parent.mkdir()
    path.write_bytes(b"[]")

    def fail_replace(self, target):
        raise PermissionError("test replace failure")

    monkeypatch.setattr(type(path), "replace", fail_replace)
    assert scraper.main([]) == 1
    assert path.read_bytes() == b"[]"
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("content", [b"{", b"{}", b'[null]'])
def test_invalid_canonical_data_stops_before_client(run, monkeypatch, caplog, content):
    _, path = run
    path.parent.mkdir()
    path.write_bytes(content)
    monkeypatch.setattr(scraper, "HLTVClient", lambda: pytest.fail("client initialized"))
    assert scraper.main([]) == 1
    assert path.read_bytes() == content
    assert "Repair or restore" in caplog.text


def test_results_parser_keeps_matches_with_missing_date(monkeypatch):
    class Driver:
        page_source = """
          <div class="results-sublist"><a class="a-reset" href="/matches/42/example">
          <div class="team1">G2</div><div class="team2">Other</div></a></div>
        """

        def get(self, url):
            pass

    client = HLTVClient()
    client.driver = Driver()
    monkeypatch.setattr(client, "_wait_for_cloudflare", lambda: None)
    [row] = client.fetch_recent_results()
    assert row["date"] is None
    assert row["url"] == "https://www.hltv.org/matches/42/example"
    client.driver.page_source = "<html><title>Unexpected page</title></html>"
    with pytest.raises(RuntimeError, match="Unrecognizable"):
        client.fetch_recent_results()
