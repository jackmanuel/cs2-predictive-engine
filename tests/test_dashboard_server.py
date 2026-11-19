import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

from dashboard_server import DashboardHandler, render_team_placeholder_svg


def test_render_team_placeholder_svg_normalizes_and_escapes():
    svg_lower = render_team_placeholder_svg("e")
    assert svg_lower.startswith("<svg ")
    assert ">E</text>" in svg_lower

    svg_empty = render_team_placeholder_svg("")
    assert ">?</text>" in svg_empty

    svg_escaped = render_team_placeholder_svg("<script>")
    assert ">&lt;</text>" in svg_escaped
    assert "<script>" not in svg_escaped


def test_dashboard_serves_dynamic_team_placeholder_svg_and_favicon():
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/dynamic-svg/teamplaceholder?letter=s") as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("image/svg+xml")
            body = resp.read().decode("utf-8")
            assert ">S</text>" in body

        with urlopen(f"http://127.0.0.1:{port}/dynamic-svg/teamplaceholder") as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
            assert ">?</text>" in body

        with urlopen(f"http://127.0.0.1:{port}/favicon.ico") as resp:
            assert resp.status == 200
            assert len(resp.read()) > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
