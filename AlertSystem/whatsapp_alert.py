from pathlib import Path
from dotenv import load_dotenv
import os 
load_dotenv()


    
from pathlib import Path
import requests

class WhatsAPPAlert:
    """
    Sends WhatsApp alerts silently in the background using an API gateway.
    No GUI display or browser required!
    """
    def __init__(self, whatsApp_num: str, message: str | Path):
        self.number = whatsApp_num
        self.message = str(message)
        self.api = os.getenv("WHATSAPP_API")
        # Example using UltraMsg instance (Replace with your actual instance/token)
        self.instance_id = "instance189857"
        self.api_url = f"https://api.ultramsg.com/{self.instance_id}/messages/chat"
        self.token = self.api

    def send(self):
        payload = {
            "token": self.token,
            "to": self.number,
            "body": self.message
        }
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        
        try:
            response = requests.post(self.api_url, data=payload, headers=headers, timeout=10)
            response.raise_for_status()
            return {"status": "success", "response": response.json()}
        except requests.exceptions.RequestException as e:
            return {"status": "failed", "error": str(e)}

if __name__ == '__main__':
    # Initialize and send without blocking your CCTV camera stream
    mobile = "+919310905797"
    mesage = "⚠️ Alert: Intrusion detected in Zone A!"
    wn = WhatsAPPAlert(mobile, mesage)
    print(wn.send())




