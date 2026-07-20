"""Local-only HTTP API and static file server for the control panel."""

from __future__ import annotations

import json
import logging
import mimetypes
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .configuration import ConfigurationService
from .jobs import JobManager
from .registry import command_catalog, command_map
from .repository import RepositoryService

logger = logging.getLogger(__name__)
STATIC_ROOT = Path(__file__).parent / "static"
MAX_REQUEST_BYTES = 2_000_000


class ControlPanelApplication:
    """Compose repository services behind a small JSON API."""

    def __init__(
        self,
        repository_root: Path,
        state_root: Path,
        python_executable: Path,
        token: str | None = None,
    ):
        self.repository_root = repository_root.resolve()
        self.state_root = state_root.resolve()
        self.token = token or secrets.token_urlsafe(32)
        self.commands = command_map()
        self.jobs = JobManager(
            self.repository_root,
            self.state_root,
            self.commands,
            python_executable,
        )
        self.configurations = ConfigurationService(
            self.repository_root,
            self.state_root,
        )
        sources = {command.source for command in self.commands.values()}
        self.repository = RepositoryService(self.repository_root, sources)

    def bootstrap(self) -> dict[str, Any]:
        """Return all initial navigation metadata and current summary state."""
        return {
            "project": {
                "name": "Regicide AI",
                "root": str(self.repository_root),
            },
            "commands": [command.as_dict() for command in command_catalog()],
            "configurations": self.configurations.definitions(),
            "scopes": [
                {"id": "artifacts", "label": "Artifacts"},
                {"id": "docs", "label": "Documentation"},
                {"id": "rules", "label": "Rules"},
                {"id": "papers", "label": "Papers"},
                {"id": "repo", "label": "Repository"},
            ],
            "overview": self._overview(),
            "jobs": self.jobs.list_jobs(50),
            "runs": self.repository.list_runs(50),
        }

    def get_api(self, route: str, query: dict[str, list[str]]) -> dict[str, Any]:
        """Dispatch a read-only API request by stable route name."""
        if route == "/api/bootstrap":
            return self.bootstrap()
        if route == "/api/overview":
            return self._overview()
        if route == "/api/jobs":
            return {"jobs": self.jobs.list_jobs(_integer_query(query, "limit", 100))}
        if route == "/api/job-log":
            return self.jobs.tail(_required_query(query, "id"))
        if route == "/api/runs":
            return {
                "runs": self.repository.list_runs(
                    limit=_integer_query(query, "limit", 100),
                    status=_optional_query(query, "status"),
                    run_type=_optional_query(query, "type"),
                    query=_optional_query(query, "q"),
                )
            }
        if route == "/api/run":
            return self.repository.run_detail(_required_query(query, "id"))
        if route == "/api/game":
            return self.repository.game_detail(_required_query(query, "id"))
        if route == "/api/config":
            return self.configurations.read(_required_query(query, "id"))
        if route == "/api/browser":
            return self.repository.list_directory(
                _required_query(query, "scope"),
                _optional_query(query, "path") or "",
            )
        if route == "/api/text":
            return self.repository.read_text_file(
                _required_query(query, "scope"),
                _required_query(query, "path"),
            )
        raise KeyError(f"Unknown API route: {route}")

    def post_api(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a validated state-changing API request."""
        if route == "/api/jobs/start":
            return self.jobs.start(
                str(payload.get("command_id", "")),
                payload.get("parameters") or {},
            )
        if route == "/api/jobs/stop":
            return self.jobs.stop(
                str(payload.get("job_id", "")),
                force=bool(payload.get("force", False)),
            )
        if route == "/api/config/preview":
            return self.configurations.preview(
                str(payload.get("config_id", "")),
                str(payload.get("text", "")),
                payload.get("expected_sha256"),
            )
        if route == "/api/config/save":
            return self.configurations.save(
                str(payload.get("config_id", "")),
                str(payload.get("text", "")),
                str(payload.get("expected_sha256", "")),
            )
        raise KeyError(f"Unknown API route: {route}")

    def _overview(self) -> dict[str, Any]:
        """Merge repository state with the panel-owned process registry."""
        overview = self.repository.overview()
        jobs = self.jobs.list_jobs(25)
        overview["jobs"] = {
            "active": sum(job["active"] for job in jobs),
            "recent": jobs,
        }
        return overview


class ControlPanelServer(ThreadingHTTPServer):
    """Threaded localhost server carrying the composed application services."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: ControlPanelApplication,
    ):
        self.application = application
        super().__init__(server_address, ControlPanelHandler)


class ControlPanelHandler(BaseHTTPRequestHandler):
    """Serve static assets, authenticated JSON routes, and approved file views."""

    server: ControlPanelServer

    def do_GET(self) -> None:
        """Handle static, API, and approved viewer requests."""
        try:
            self._require_local_host()
        except PermissionError as error:
            self._send_api_error(error)
            return
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api_get(parsed)
            return
        if parsed.path == "/view":
            self._handle_view(parsed)
            return
        if parsed.path in {"/", "/index.html"}:
            self._serve_index()
            return
        if parsed.path.startswith("/static/"):
            self._serve_static(parsed.path.removeprefix("/static/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Handle authenticated JSON mutations and panel shutdown."""
        try:
            self._require_local_host()
        except PermissionError as error:
            self._send_api_error(error)
            return
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            self._require_token()
            payload = self._read_json_body()
            if parsed.path == "/api/shutdown":
                self._send_json({"status": "shutting-down"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            response = self.server.application.post_api(parsed.path, payload)
            self._send_json(response)
        except Exception as error:  # noqa: BLE001 - API boundary maps errors.
            self._send_api_error(error)

    def log_message(self, format_string: str, *arguments: Any) -> None:
        """Route HTTP access messages into the panel's rotating file log."""
        logger.debug("HTTP %s - %s", self.address_string(), format_string % arguments)

    def _handle_api_get(self, parsed: Any) -> None:
        """Authenticate and dispatch one JSON GET request."""
        try:
            self._require_token()
            response = self.server.application.get_api(
                parsed.path,
                parse_qs(parsed.query, keep_blank_values=True),
            )
            self._send_json(response)
        except Exception as error:  # noqa: BLE001 - API boundary maps errors.
            self._send_api_error(error)

    def _handle_view(self, parsed: Any) -> None:
        """Serve one approved file with restrictive browser headers."""
        query = parse_qs(parsed.query, keep_blank_values=True)
        try:
            token = _required_query(query, "token")
            if not secrets.compare_digest(token, self.server.application.token):
                raise PermissionError("Invalid viewer token")
            scope = _required_query(query, "scope")
            relative_path = _required_query(query, "path")
            path = self.server.application.repository.resolve_view_file(
                scope,
                relative_path,
            )
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            disposition = "attachment" if _optional_query(query, "download") else "inline"
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'{disposition}; filename="{path.name}"')
            self.send_header("X-Content-Type-Options", "nosniff")
            if path.suffix.lower() == ".html":
                self.send_header(
                    "Content-Security-Policy",
                    "sandbox; default-src 'self' data: blob:; img-src 'self' data: blob:; "
                    "style-src 'self' 'unsafe-inline'",
                )
            self.end_headers()
            self.wfile.write(data)
        except Exception as error:  # noqa: BLE001 - viewer boundary maps errors.
            self._send_api_error(error)

    def _serve_index(self) -> None:
        """Inject the ephemeral API token into the application shell."""
        template = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        content = template.replace("__CONTROL_PANEL_TOKEN__", self.server.application.token)
        self._send_bytes(content.encode("utf-8"), "text/html; charset=utf-8")

    def _serve_static(self, relative_path: str) -> None:
        """Serve a regular file contained by the package static directory."""
        path = (STATIC_ROOT / relative_path).resolve(strict=False)
        try:
            path.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or path.suffix in {".js", ".css"}:
            content_type += "; charset=utf-8"
        self._send_bytes(path.read_bytes(), content_type, cache_seconds=3600)

    def _require_token(self) -> None:
        """Reject API access without the page's ephemeral header token."""
        supplied = self.headers.get("X-Control-Token", "")
        if not secrets.compare_digest(supplied, self.server.application.token):
            raise PermissionError("Invalid control-panel token")

    def _require_local_host(self) -> None:
        """Reject DNS-rebinding requests whose Host is not loopback-local."""
        host_header = self.headers.get("Host", "")
        hostname = urlparse(f"//{host_header}").hostname
        if hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise PermissionError("Control panel accepts only localhost requests")

    def _read_json_body(self) -> dict[str, Any]:
        """Read a bounded JSON object and reject cross-origin form requests."""
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError("Mutations require application/json")
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request body is too large")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        """Serialize a JSON response with no-store security headers."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        cache_seconds: int = 0,
    ) -> None:
        """Send a static response with a restrictive application CSP."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header(
            "Cache-Control",
            f"public, max-age={cache_seconds}" if cache_seconds else "no-store",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        if content_type.startswith("text/html"):
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; frame-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
        self.end_headers()
        self.wfile.write(data)

    def _send_api_error(self, error: Exception) -> None:
        """Map expected service exceptions to concise JSON status responses."""
        if isinstance(error, (KeyError, FileNotFoundError, NotADirectoryError)):
            status = HTTPStatus.NOT_FOUND
        elif isinstance(error, PermissionError):
            status = HTTPStatus.FORBIDDEN
        elif isinstance(error, RuntimeError):
            status = HTTPStatus.CONFLICT
        else:
            status = HTTPStatus.BAD_REQUEST
        logger.warning("Control-panel request failed: %s", error)
        self._send_json(
            {"error": str(error).strip("'"), "type": type(error).__name__},
            status,
        )


def _required_query(query: dict[str, list[str]], name: str) -> str:
    """Return one non-empty query value or raise a readable error."""
    value = _optional_query(query, name)
    if not value:
        raise ValueError(f"Missing query parameter: {name}")
    return value


def _optional_query(query: dict[str, list[str]], name: str) -> str | None:
    """Return the first query value when present and non-empty."""
    values = query.get(name)
    if not values or values[0] == "":
        return None
    return values[0]


def _integer_query(
    query: dict[str, list[str]],
    name: str,
    default: int,
) -> int:
    """Parse a bounded positive integer query parameter."""
    raw = _optional_query(query, name)
    if raw is None:
        return default
    value = int(raw)
    return max(1, min(value, 500))
