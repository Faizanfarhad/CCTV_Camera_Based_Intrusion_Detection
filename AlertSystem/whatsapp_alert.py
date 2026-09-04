"""WhatsApp alert integrations.

WasenderAPI is the default provider. UltraMsg remains available only when
``WHATSAPP_PROVIDER=ultramsg`` is explicitly configured for compatibility.
"""

import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


class WhatsAPPAlert:
    """Send a WhatsApp text message through the configured provider."""

    def __init__(self, whatsApp_num: str, message: str | Path):
        self.number = self._normalize_recipient(whatsApp_num)
        self.message = str(message)
        self.provider = os.getenv("WHATSAPP_PROVIDER", "wasenderapi").strip().lower()

    @staticmethod
    def _normalize_recipient(value: str) -> str:
        """Keep E.164 numbers and WhatsApp JIDs usable by either provider."""
        recipient = str(value or "").strip()
        if "@" in recipient:
            return recipient
        return re.sub(r"[^0-9+]", "", recipient)

    def _send_wasender(self):
        api_key = (
            os.getenv("WASENDER_API_KEY", "").strip()
            or os.getenv("WASENDER_API_TOKEN", "").strip()
        )
        if not api_key:
            return {"status": "failed", "error": "WASENDER_API_KEY is not configured"}

        url = os.getenv(
            "WASENDER_API_URL",
            "https://www.wasenderapi.com/api/send-message",
        ).strip()
        payload = {"to": self.number, "text": self.message}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("success") is False:
            return {"status": "failed", "error": data.get("message", data)}
        return {"status": "success", "response": data}

    def _send_ultramsg(self):
        token = os.getenv("WHATSAPP_API", "").strip()
        instance_id = os.getenv("ULTRAMSG_INSTANCE_ID", "instance189857").strip()
        if not token:
            return {"status": "failed", "error": "WHATSAPP_API is not configured"}

        url = f"https://api.ultramsg.com/{instance_id}/messages/chat"
        payload = {"token": token, "to": self.number, "body": self.message}
        response = requests.post(url, data=payload, timeout=15)
        response.raise_for_status()
        return {"status": "success", "response": response.json()}

    def send(self):
        try:
            if self.provider in {"ultramsg", "ultra"}:
                return self._send_ultramsg()
            return self._send_wasender()
        except requests.exceptions.RequestException as exc:
            return {"status": "failed", "error": str(exc)}
        except (TypeError, ValueError) as exc:
            return {"status": "failed", "error": f"Invalid WhatsApp API response: {exc}"}


if __name__ == "__main__":
    mobile = "+919310905797"
    message = "Alert: Intrusion detected in the ROI."
    print(WhatsAPPAlert(mobile, message).send())


