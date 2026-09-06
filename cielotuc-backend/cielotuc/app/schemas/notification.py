"""
app/schemas/notification.py
────────────────────────────
Pydantic schemas for the notification endpoints.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class NotificationSend(BaseModel):
    """Request body for POST /api/v1/notifications/send."""
    to: str = Field(
        ...,
        description="Recipient phone in E.164 format (+549381xxxxxxx)",
        examples=["+5493815551234"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Message body",
    )
    channels: List[str] = Field(
        default=["sms"],
        description='Channels to send on: "sms", "whatsapp"',
        examples=[["sms", "whatsapp"]],
    )


class NotificationResultOut(BaseModel):
    """Single channel result."""
    channel: str
    to: str
    sent: bool
    message_sid: Optional[str] = None
    error: Optional[str] = None


class NotificationSendResponse(BaseModel):
    """Response for POST /api/v1/notifications/send."""
    status: str
    results: List[NotificationResultOut]


class NotificationAlertPreset(BaseModel):
    """Pre-defined alert template for common scenarios."""
    zone_name: str
    alert_type: str = Field(
        ...,
        description='"flood" | "zonda" | "storm"',
    )
    severity: str = Field(default="moderate")
    rain_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_precip_mm: float = Field(default=0.0, ge=0.0)
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    wind_speed_kmh: float = Field(default=0.0, ge=0.0)
    hail_risk: float = Field(default=0.0, ge=0.0, le=1.0)
