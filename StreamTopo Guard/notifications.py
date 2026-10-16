"""Opt-in, data-minimized Slack and Discord anomaly notifications."""

from __future__ import annotations

import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from typing import Any


WEBHOOK_ENV_VAR = "TOPOGUARD_WEBHOOK_URL"


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


class WebhookNotifier:
    """Deliver compact anomaly summaries to Slack or Discord incoming webhooks."""

    def __init__(self, webhook_url: str, timeout: float = 5.0):
        parsed = urlsplit(webhook_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("Webhook URL must be an HTTPS Slack or Discord endpoint")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Webhook URL must not contain basic-auth credentials")

        hostname = parsed.hostname.casefold()
        if hostname == "hooks.slack.com" and parsed.path.startswith("/services/"):
            self._platform = "slack"
        elif hostname in {"discord.com", "discordapp.com"} and parsed.path.startswith(
            "/api/webhooks/"
        ):
            self._platform = "discord"
        else:
            raise ValueError("Webhook URL must be a Slack or Discord incoming webhook")
        if timeout <= 0:
            raise ValueError("Webhook timeout must be positive")

        self._url = webhook_url
        self._timeout = timeout

    @classmethod
    def from_environment(cls) -> WebhookNotifier:
        webhook_url = os.environ.get(WEBHOOK_ENV_VAR, "").strip()
        if not webhook_url:
            raise ValueError(f"Set {WEBHOOK_ENV_VAR} to a Slack or Discord HTTPS webhook URL")
        return cls(webhook_url)

    @staticmethod
    def _is_alert(result: dict[str, Any]) -> bool:
        if result.get("is_anomaly"):
            return True
        for key in ("z3_result", "security_boundary"):
            verification = result.get(key)
            if isinstance(verification, dict) and verification.get("is_violated"):
                return True
        return False

    def send_alert(self, result: dict[str, Any], source: str, tick: int) -> bool:
        if not self._is_alert(result):
            return False

        context = result.get("context")
        if not isinstance(context, dict):
            context = {}
        try:
            score = float(result.get("anomaly_score", context.get("anomaly_score", 0.0)))
        except (TypeError, ValueError) as exc:
            raise ValueError("Alert anomaly score must be numeric") from exc
        if not math.isfinite(score):
            raise ValueError("Alert anomaly score must be finite")

        severity = str(context.get("severity", result.get("severity", "UNKNOWN")))
        message = (
            f"TopoGuard alert | source={source} tick={tick} "
            f"score={score:.4f} severity={severity}"
        )
        attack_context = context.get("attack_context") if isinstance(context, dict) else None
        if attack_context and attack_context.get("correlated"):
            summary = attack_context.get("summary", "")
            message += f" | attack_context={summary}"
        payload = {"text": message} if self._platform == "slack" else {"content": message}
        request = Request(
            self._url,
            data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            opener = build_opener(_NoRedirectHandler())
            with opener.open(request, timeout=self._timeout) as response:
                status = getattr(response, "status", 200)
                if not 200 <= status < 300:
                    raise RuntimeError(f"Webhook returned HTTP {status}")
        except HTTPError as exc:
            raise RuntimeError(f"Webhook delivery failed with HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc.__class__.__name__)
            raise RuntimeError(f"Webhook delivery failed: {reason}") from None

        return True
