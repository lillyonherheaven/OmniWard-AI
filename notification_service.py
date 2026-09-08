"""
Smart Visual-Acoustic Patient Monitor - Emergency Notification Service
Handles Level 3 critical escalations via Twilio SMS and Hospital Push Webhooks.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import requests

from settings import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    EMERGENCY_DESK_PHONE,
    PUSH_NOTIFICATION_WEBHOOK,
)

# Configure logger
logger = logging.getLogger("patient_monitor.notifications")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [NOTIFICATION SERVICE] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class NotificationService:
    """
    Manages urgent clinical escalation alerts when patient safety is critically compromised.
    Provides simulated and live omnichannel dispatches (SMS, Push Webhooks, and Central Station Beacon).
    """

    def __init__(self):
        self.twilio_sid = TWILIO_ACCOUNT_SID
        self.twilio_token = TWILIO_AUTH_TOKEN
        self.twilio_from = TWILIO_PHONE_NUMBER
        self.emergency_contact = EMERGENCY_DESK_PHONE
        self.webhook_url = PUSH_NOTIFICATION_WEBHOOK
        self.dispatch_history: List[Dict[str, Any]] = []

    def send_critical_alert(
        self,
        room_id: str,
        patient_id: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dispatches Level 3 emergency code blue escalation alert.
        Simulates / executes Twilio SMS and Hospital Webhook alerts.
        """
        timestamp = datetime.now().isoformat()
        alert_payload = {
            "timestamp": timestamp,
            "severity": "CRITICAL (Level 3)",
            "room_id": room_id,
            "patient_id": patient_id,
            "event_type": event_type,
            "details": details or {},
            "urgency": "IMMEDIATE_RESPONSE_REQUIRED",
            "message": (
                f"🚨 [CODE BLUE ALERT] Patient {patient_id} in Room {room_id} "
                f"triggered critical event '{event_type.upper()}'. Immediate nurse intervention required!"
            )
        }

        # 1. Twilio SMS Escalation
        sms_status = self._dispatch_twilio_sms(alert_payload)

        # 2. Push Webhook Escalation
        webhook_status = self._dispatch_webhook(alert_payload)

        record = {
            "alert_id": f"ALT-{int(datetime.now().timestamp() * 1000)}",
            "timestamp": timestamp,
            "room_id": room_id,
            "patient_id": patient_id,
            "event_type": event_type,
            "sms_dispatch": sms_status,
            "webhook_dispatch": webhook_status,
            "status": "DISPATCHED"
        }

        self.dispatch_history.append(record)
        logger.warning(
            f"CRITICAL ESCALATION DISPATCHED -> Room: {room_id} | Patient: {patient_id} | "
            f"Event: {event_type} | SMS: {sms_status['status']} | Webhook: {webhook_status['status']}"
        )
        return record

    def _dispatch_twilio_sms(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sends SMS via Twilio API or provides simulated telemetry dispatch."""
        is_live_configured = (
            bool(self.twilio_sid) and
            not self.twilio_sid.startswith("AC_MOCK") and
            bool(self.twilio_token) and
            not self.twilio_token.startswith("MOCK")
        )

        if is_live_configured:
            try:
                url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_sid}/Messages.json"
                auth = (self.twilio_sid, self.twilio_token)
                data = {
                    "From": self.twilio_from,
                    "To": self.emergency_contact,
                    "Body": payload["message"]
                }
                resp = requests.post(url, data=data, auth=auth, timeout=5.0)
                if resp.status_code in [200, 201]:
                    return {"status": "DELIVERED", "channel": "Twilio SMS", "to": self.emergency_contact}
                else:
                    return {"status": "FAILED", "code": resp.status_code, "channel": "Twilio SMS"}
            except Exception as e:
                logger.error(f"Failed to transmit live Twilio SMS: {e}")
                return {"status": "ERROR", "error": str(e), "channel": "Twilio SMS"}

        # Simulation Mode
        return {
            "status": "SIMULATED_SUCCESS",
            "channel": "Twilio SMS Gateway",
            "recipient": self.emergency_contact,
            "message": payload["message"]
        }

    def _dispatch_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Posts urgent alert payload to configured medical dispatch webhook."""
        if self.webhook_url and not self.webhook_url.startswith("https://hospital.emergency.internal"):
            try:
                resp = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=4.0
                )
                return {
                    "status": "DELIVERED" if resp.status_code < 400 else "REJECTED",
                    "status_code": resp.status_code,
                    "channel": "Hospital Push Webhook"
                }
            except Exception as e:
                logger.error(f"Webhook connection error: {e}")
                return {"status": "NETWORK_FAILURE", "error": str(e), "channel": "Hospital Push Webhook"}

        # Simulation Mode
        return {
            "status": "SIMULATED_SUCCESS",
            "channel": "Hospital Intranet Push Webhook",
            "target": self.webhook_url or "Internal Central Station Broadcast"
        }

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent critical escalation dispatches."""
        return self.dispatch_history[-limit:]
