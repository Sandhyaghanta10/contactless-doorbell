"""Email alerts. Secrets come from environment variables, never from code."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Callable, Protocol


class Notifier(Protocol):
    def notify(self, subject: str, body: str) -> None: ...


class PrintNotifier:
    """Dry-run notifier: prints instead of sending."""

    def notify(self, subject: str, body: str) -> None:
        print(f"[ALERT] {subject}: {body}")


def build_message(sender: str, recipient: str, subject: str, body: str) -> EmailMessage:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, recipient, subject
    msg.set_content(body)
    return msg


class EmailNotifier:
    def __init__(self, host: str, port: int, user: str, password: str, recipient: str,
                 smtp_factory: Callable[..., smtplib.SMTP] = smtplib.SMTP) -> None:
        self._host, self._port, self._user, self._password = host, port, user, password
        self._recipient, self._smtp_factory = recipient, smtp_factory

    @classmethod
    def from_env(cls) -> "EmailNotifier":
        try:
            return cls(os.environ["DOORBELL_SMTP_HOST"], int(os.environ.get("DOORBELL_SMTP_PORT", "587")),
                       os.environ["DOORBELL_SMTP_USER"], os.environ["DOORBELL_SMTP_PASSWORD"],
                       os.environ["DOORBELL_NOTIFY_TO"])
        except KeyError as missing:
            raise SystemExit(f"Set {missing.args[0]} (or use --dry-run)")

    def notify(self, subject: str, body: str) -> None:
        msg = build_message(self._user, self._recipient, subject, body)
        with self._smtp_factory(self._host, self._port) as smtp:
            smtp.starttls()
            smtp.login(self._user, self._password)
            smtp.send_message(msg)
