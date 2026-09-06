"""
app/services/notification_service.py
─────────────────────────────────────
Sends weather alerts via Twilio SMS and WhatsApp.

Used by:
  - FloodAlertService (automatic alerts)
  - POST /api/v1/notifications/send (manual dashboard alerts)
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from app.core.config import settings


@dataclass
class NotificationResult:
    channel: str          # "sms" | "whatsapp"
    to: str
    sent: bool
    message_sid: Optional[str] = None
    error: Optional[str] = None


class NotificationService:
    """
    Thin wrapper around Twilio REST client.

    Lazy-initializes the Twilio client on first use so the app
    can start even without Twilio credentials configured.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not settings.twilio_account_sid or not settings.twilio_auth_token:
                raise RuntimeError(
                    "Twilio not configured — set TWILIO_ACCOUNT_SID "
                    "and TWILIO_AUTH_TOKEN in .env"
                )
            from twilio.rest import Client
            self._client = Client(
                settings.twilio_account_sid,
                settings.twilio_auth_token,
            )
        return self._client

    # ── SMS ────────────────────────────────────────────────────

    def send_sms(
        self,
        to: str,
        body: str,
    ) -> NotificationResult:
        """
        Send an SMS alert.

        Args:
            to: Recipient phone number in E.164 format (+549381xxxxxxx)
            body: Message body (max 1600 chars for SMS)
        """
        if not settings.twilio_from_phone:
            return NotificationResult(
                channel="sms", to=to, sent=False,
                error="TWILIO_FROM_PHONE not configured",
            )

        try:
            client = self._get_client()
            message = client.messages.create(
                body=body[:1600],
                from_=settings.twilio_from_phone,
                to=to,
            )
            logger.info(f"SMS sent → {to} SID={message.sid}")
            return NotificationResult(
                channel="sms", to=to, sent=True,
                message_sid=message.sid,
            )
        except Exception as e:
            logger.error(f"SMS failed → {to}: {e}")
            return NotificationResult(
                channel="sms", to=to, sent=False, error=str(e),
            )

    # ── WhatsApp ───────────────────────────────────────────────

    def send_whatsapp(
        self,
        to: str,
        body: str,
    ) -> NotificationResult:
        """
        Send a WhatsApp alert via Twilio Sandbox.

        Args:
            to: Recipient WhatsApp number in E.164 format
                (must have joined the Twilio Sandbox first)
            body: Message body (max 4096 chars for WhatsApp)
        """
        if not settings.twilio_whatsapp_from:
            return NotificationResult(
                channel="whatsapp", to=to, sent=False,
                error="TWILIO_WHATSAPP_FROM not configured",
            )

        try:
            client = self._get_client()
            from_number = f"whatsapp:{settings.twilio_whatsapp_from}"
            to_number = f"whatsapp:{to}" if not to.startswith("whatsapp:") else to

            message = client.messages.create(
                body=body[:4096],
                from_=from_number,
                to=to_number,
            )
            logger.info(f"WhatsApp sent → {to} SID={message.sid}")
            return NotificationResult(
                channel="whatsapp", to=to, sent=True,
                message_sid=message.sid,
            )
        except Exception as e:
            logger.error(f"WhatsApp failed → {to}: {e}")
            return NotificationResult(
                channel="whatsapp", to=to, sent=False, error=str(e),
            )

    # ── Multi-channel ──────────────────────────────────────────

    def send_alert(
        self,
        to: str,
        body: str,
        channels: list[str] | None = None,
    ) -> list[NotificationResult]:
        """
        Send an alert to one or more channels.

        Args:
            to: Recipient phone number (E.164)
            body: Message body
            channels: List of channels ("sms", "whatsapp").
                      Defaults to ["sms"].
        """
        channels = channels or ["sms"]
        results = []

        for ch in channels:
            if ch == "sms":
                results.append(self.send_sms(to, body))
            elif ch == "whatsapp":
                results.append(self.send_whatsapp(to, body))
            else:
                results.append(NotificationResult(
                    channel=ch, to=to, sent=False,
                    error=f"Unknown channel: {ch}",
                ))

        return results

    # ── Message templates ──────────────────────────────────────

    @staticmethod
    def format_flood_alert(
        zone_name: str,
        severity: str,
        rain_probability: float,
        expected_precip_mm: float,
    ) -> str:
        """Format a flood alert message for SMS/WhatsApp."""
        severity_es = {
            "preventive": "Preventivo",
            "moderate": "Moderado",
            "critical": "CRITICO",
        }.get(severity, severity)

        emoji = {
            "preventive": "🟡",
            "moderate": "🟠",
            "critical": "🔴",
        }.get(severity, "⚠️")

        return (
            f"{emoji} CIELO·TUC — Alerta {severity_es}\n\n"
            f"Zona: {zone_name}\n"
            f"Probabilidad de lluvia: {rain_probability:.0%}\n"
            f"Precipitación esperada: {expected_precip_mm:.1f} mm\n\n"
            f"Monitorice las condiciones. Ante lluvias intensas, "
            f"evite zonas inundables y cercanías de ríos."
        )

    @staticmethod
    def format_zonda_alert(
        zone_name: str,
        risk_score: float,
        wind_speed_kmh: float,
    ) -> str:
        """Format a Zonda wind alert message."""
        return (
            f"💨 CIELO·TUC — Alerta Zonda\n\n"
            f"Zona: {zone_name}\n"
            f"Riesgo: {risk_score:.0f}/100\n"
            f"Velocidad del viento: {wind_speed_kmh:.0f} km/h\n\n"
            f"Viento Zonda detectado/previsto. Cierre ventanas, "
            f"cuidado con objetos sueltos y vegetación seca."
        )

    @staticmethod
    def format_storm_alert(
        zone_name: str,
        rain_probability: float,
        hail_risk: float,
    ) -> str:
        """Format a severe storm alert message."""
        hail_text = " y granizo probable" if hail_risk > 0.5 else ""
        return (
            f"⛈ CIELO·TUC — Tormenta Severa\n\n"
            f"Zona: {zone_name}\n"
            f"Probabilidad de lluvia: {rain_probability:.0%}{hail_text}\n\n"
            f"Busque refugio. Evite áreas abiertas y zones elevadas."
        )
