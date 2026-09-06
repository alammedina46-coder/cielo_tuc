"""
app/services/flood_alert_service.py
────────────────────────────────────
Sends automatic alerts to FLOOD·TUC when CIELO·TUC predicts
extreme rainfall (>= settings.floodtuc_alert_threshold_mm).

Also handles manual alerts triggered from the government dashboard.
Sends SMS/WhatsApp notifications to emergency contacts after alert.
"""
from datetime import datetime
from typing import Optional

import httpx
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.weather import FloodAlert, Zone
from app.schemas.weather import FloodAlertCreate, FloodAlertOut


class FloodAlertService:
    """
    Thin HTTP client wrapper + DB persistence for FLOOD·TUC alerts.
    """

    def __init__(self):
        self._notification_service = None

    def _get_notification_service(self):
        if self._notification_service is None:
            from app.services.notification_service import NotificationService
            self._notification_service = NotificationService()
        return self._notification_service

    async def send_alert(
        self,
        zone_id: int,
        rain_probability: float,
        expected_precip_mm: float,
        db: AsyncSession,
        severity: str | None = None,
        prediction_id: int | None = None,
        trigger_type: str = "automatic",
        notes: Optional[str] = None,
    ) -> FloodAlertOut:
        """
        Dispatch an alert to FLOOD·TUC and persist it in the DB.

        Severity is auto-derived from expected precipitation if not
        explicitly provided:
          < 70mm  -> preventive
          70-120mm -> moderate
          > 120mm  -> critical
        """
        if severity is None:
            if expected_precip_mm < 70:
                severity = "preventive"
            elif expected_precip_mm <= 120:
                severity = "moderate"
            else:
                severity = "critical"

        # -- Build FLOOD·TUC payload --
        zone = await db.get(Zone, zone_id)
        zone_name = zone.name if zone else str(zone_id)
        payload = {
            "source": "CIELO·TUC",
            "zone_id": zone_id,
            "zone_name": zone_name,
            "severity": severity,
            "rain_probability": round(rain_probability, 3),
            "expected_precip_mm": round(expected_precip_mm, 1),
            "trigger_type": trigger_type,
            "generated_at": datetime.utcnow().isoformat(),
            "notes": notes,
        }

        # -- Send HTTP request --
        status_code = None
        response_body = None
        delivered = False

        if settings.floodtuc_api_url and settings.floodtuc_api_key:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        f"{settings.floodtuc_api_url}/api/v1/alerts/inbound",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {settings.floodtuc_api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    status_code = resp.status_code
                    response_body = resp.text[:500]
                    delivered = resp.status_code < 300
                    logger.info(
                        f"FLOOD·TUC alert sent -> zone {zone_id} "
                        f"severity={severity} HTTP {status_code}"
                    )
            except Exception as e:
                logger.error(f"FLOOD·TUC alert failed: {e}")
                response_body = str(e)[:500]
        else:
            logger.warning(
                "FLOOD·TUC not configured -- alert logged but not sent"
            )
            delivered = False

        # -- Persist in DB --
        alert = FloodAlert(
            zone_id=zone_id,
            prediction_id=prediction_id,
            trigger_type=trigger_type,
            rain_probability=rain_probability,
            expected_precip_mm=expected_precip_mm,
            severity=severity,
            floodtuc_response_code=status_code,
            floodtuc_response_body=response_body,
            delivered=delivered,
        )
        db.add(alert)
        await db.flush()

        # -- Send SMS/WhatsApp notifications --
        self._send_notifications(
            zone_name=zone_name,
            severity=severity,
            rain_probability=rain_probability,
            expected_precip_mm=expected_precip_mm,
        )

        return FloodAlertOut(
            id=alert.id,
            zone_id=zone_id,
            severity=severity,
            rain_probability=rain_probability,
            expected_precip_mm=expected_precip_mm,
            trigger_type=trigger_type,
            triggered_at=alert.triggered_at or datetime.utcnow(),
            delivered=delivered,
            floodtuc_response_code=status_code,
            notes=notes,
        )

    def _send_notifications(
        self,
        zone_name: str,
        severity: str,
        rain_probability: float,
        expected_precip_mm: float,
    ) -> None:
        """
        Best-effort SMS/WhatsApp notification to emergency contacts.
        Runs synchronously -- failures are logged but do not affect
        the alert flow.
        """
        recipients = settings.twilio_alert_recipients
        if not recipients:
            return

        try:
            ns = self._get_notification_service()
            body = ns.format_flood_alert(
                zone_name=zone_name,
                severity=severity,
                rain_probability=rain_probability,
                expected_precip_mm=expected_precip_mm,
            )
            for phone in recipients:
                result = ns.send_sms(to=phone, body=body)
                if result.sent:
                    logger.info(f"SMS alert sent -> {phone} zone={zone_name}")
                else:
                    logger.warning(f"SMS failed -> {phone}: {result.error}")
        except Exception as e:
            logger.error(f"Notification dispatch failed: {e}")
