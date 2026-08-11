"""In-process mock of the LumaBot daemon.

LumaKit's robot tools talk HTTP to `LUMABOT_URL` (default 127.0.0.1:8971).
Trajectory episodes must exercise those tools without a robot, and — more
importantly — must be able to produce the daemon's *real failure modes*, since
error-recovery episodes are only worth generating if the failures are the ones
the hardware actually returns.

Routes and status codes mirror LumaBot's `server.py` at 7bb0d9a:

    GET  /status              -> 200 status payload
    POST /drive               -> 202 accepted | 400 bad input
                                 | 409 MotorsNotReady | 409 ObstacleSafetyError
    POST /stop                -> 200
    POST /camera/capture      -> 201 captured | 409 CameraBusy
                                 | 503 CameraUnavailable | 502 capture failed
    POST /autonomy            -> 202 accepted | 200 stopped | 400 | 409 unavailable
    POST /indicator/activity  -> 200 | 400

`fault` selects an injected failure so a seed task can request one
deterministically rather than hoping for a flake.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

FAULTS = (
    "none",
    "obstacle",          # 409 on /drive — obstacle safety
    "motors_not_ready",  # 409 on /drive
    "camera_unavailable",# 503 on /camera/capture
    "camera_busy",       # 409 on /camera/capture
    "autonomy_unavailable",  # 409 on /autonomy
    "unreachable",       # server refuses connections entirely
)

_STATE = {"fault": "none", "moving": False, "autonomy": False}


def set_fault(fault: str) -> None:
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; expected one of {FAULTS}")
    _STATE["fault"] = fault


def _status() -> dict:
    return {
        "battery_pct": 74,
        "charging": False,
        "motors_ready": _STATE["fault"] != "motors_not_ready",
        "distance_sensor_ready": _STATE["fault"] != "obstacle",
        # Deliberately false: the system prompt tells the model never to claim
        # obstacle protection is active when the fields say otherwise, and we
        # want episodes that exercise that.
        "obstacle_protection_active": False,
        "autonomy_active": _STATE["autonomy"],
        "moving": _STATE["moving"],
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep episode stdout clean
        return

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path != "/status":
            self.send_error(404)
            return
        self._send(200, _status())

    def do_POST(self):
        fault = _STATE["fault"]
        data = self._read_json()

        if self.path == "/drive":
            direction = data.get("direction")
            if not direction:
                self._send(400, {"error": "direction is required"})
                return
            if fault == "motors_not_ready":
                self._send(409, {"error": "motors are not ready"})
                return
            if fault == "obstacle":
                self._send(409, {"error": "obstacle detected within safety distance"})
                return
            _STATE["moving"] = True
            self._send(202, {"accepted": True, "direction": direction,
                             "entire_request_scheduled": True,
                             "status": _status()})
            return

        if self.path == "/stop":
            _STATE["moving"] = False
            _STATE["autonomy"] = False
            self._send(200, {"stopped": True, "status": _status()})
            return

        if self.path == "/camera/capture":
            if fault == "camera_unavailable":
                self._send(503, {"error": "camera unavailable"})
                return
            if fault == "camera_busy":
                self._send(409, {"error": "camera is busy"})
                return
            self._send(201, {"captured": True, "path": "photos/capture_0001.jpg"})
            return

        if self.path == "/autonomy":
            active = data.get("active")
            if not isinstance(active, bool):
                self._send(400, {"error": "active must be true or false"})
                return
            if active:
                if fault == "autonomy_unavailable":
                    self._send(409, {"error": "autonomy unavailable: distance "
                                              "sensor not ready"})
                    return
                _STATE["autonomy"] = True
                self._send(202, {"accepted": True, "status": _status()})
            else:
                _STATE["autonomy"] = False
                self._send(200, {"stopped": True, "status": _status()})
            return

        if self.path == "/indicator/activity":
            if not isinstance(data.get("lease_id"), str):
                self._send(400, {"error": "lease_id is required"})
                return
            self._send(200, {"ok": True})
            return

        self._send(404, {"error": "not found"})


class MockDaemon:
    """Context manager returning the base URL to put in LUMABOT_URL."""

    def __init__(self, fault: str = "none"):
        self.fault = fault
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> str | None:
        set_fault(self.fault)
        _STATE["moving"] = False
        _STATE["autonomy"] = False
        if self.fault == "unreachable":
            # Nothing listening: exercises the client's connection-error path,
            # where the correct behaviour is to report it rather than invent a
            # battery percentage.
            return "http://127.0.0.1:9"
        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, *exc) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


if __name__ == "__main__":
    import urllib.request
    for fault in ("none", "obstacle", "camera_unavailable"):
        with MockDaemon(fault) as url:
            status = json.loads(urllib.request.urlopen(f"{url}/status", timeout=5).read())
            req = urllib.request.Request(
                f"{url}/drive", data=json.dumps({"direction": "forward",
                                                 "speed": 40, "duration_s": 2}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                resp = urllib.request.urlopen(req, timeout=5)
                drive = f"{resp.status} {json.loads(resp.read()).get('accepted')}"
            except Exception as e:
                drive = f"{getattr(e, 'code', '?')} {getattr(e, 'reason', e)}"
            print(f"  fault={fault:<20} motors_ready={status['motors_ready']} "
                  f"drive -> {drive}")
