"""
app/api/v1/endpoints/notifications.py
───────────────────────────────────────
Twilio SMS / WhatsApp notification endpoints.

Requires auth (tecnico or admin) for all endpoints.
"""

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user, require_role
from app.models.user import User
from app.schemas.notification import (
    NotificationAlertPreset,
    NotificationResultOut,
    NotificationSend,
    NotificationSendResponse,
)
from app.services.notification_service import NotificationService

notifications_router = APIRouter(prefix="/notifications", tags=["Notifications"])
_notification_service = NotificationService()


@notifications_router.post(
    "/send",
    response_model=NotificationSendResponse,
    status_code=200,
)
async def send_notification(
    payload: NotificationSend,
    _user: Annotated[User, Depends(require_role("tecnico", "admin"))] = None,
):
    """
    Send a custom alert message via SMS and/or WhatsApp.

    Requires **tecnico** or **admin** role.
    """
    channels = [ch.lower() for ch in payload.channels]
    valid = {"sms", "whatsapp"}
    for ch in channels:
        if ch not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid channel '{ch}'. Use 'sms' or 'whatsapp'.",
            )

    results = _notification_service.send_alert(
        to=payload.to,
        body=payload.message,
        channels=channels,
    )

    return NotificationSendResponse(
        status="ok",
        results=[
            NotificationResultOut(
                channel=r.channel,
                to=r.to,
                sent=r.sent,
                message_sid=r.message_sid,
                error=r.error,
            )
            for r in results
        ],
    )


@notifications_router.post(
    "/alert",
    response_model=NotificationSendResponse,
    status_code=200,
)
async def send_alert_preset(
    payload: NotificationAlertPreset,
    to: str,
    channels: List[str] = ["sms"],
    _user: Annotated[User, Depends(require_role("tecnico", "admin"))] = None,
):
    """
    Send a pre-formatted weather alert using built-in templates.

    **Query params:**
    - `to`: recipient phone (E.164)
    - `channels`: comma-separated list (default: "sms")

    **Body `alert_type`:**
    - `"flood"` — Flood/lluvia alert
    - `"zonda"` — Zonda wind alert
    - `"storm"` — Severe storm alert

    Requires **tecnico** or **admin** role.
    """
    ns = NotificationService

    if payload.alert_type == "flood":
        body = ns.format_flood_alert(
            zone_name=payload.zone_name,
            severity=payload.severity,
            rain_probability=payload.rain_probability,
            expected_precip_mm=payload.expected_precip_mm,
        )
    elif payload.alert_type == "zonda":
        body = ns.format_zonda_alert(
            zone_name=payload.zone_name,
            risk_score=payload.risk_score,
            wind_speed_kmh=payload.wind_speed_kmh,
        )
    elif payload.alert_type == "storm":
        body = ns.format_storm_alert(
            zone_name=payload.zone_name,
            rain_probability=payload.rain_probability,
            hail_risk=payload.hail_risk,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown alert_type '{payload.alert_type}'. "
                   "Use 'flood', 'zonda', or 'storm'.",
        )

    ch = [c.lower() for c in channels]
    results = _notification_service.send_alert(
        to=to,
        body=body,
        channels=ch,
    )

    return NotificationSendResponse(
        status="ok",
        results=[
            NotificationResultOut(
                channel=r.channel,
                to=r.to,
                sent=r.sent,
                message_sid=r.message_sid,
                error=r.error,
            )
            for r in results
        ],
    )
