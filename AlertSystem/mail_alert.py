import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os 
import resend



load_dotenv()

class MailAlert:
    def __init__(self):
        super().__init__()
        self.RESEND_API = os.getenv("RESEND_API")
        self.response = None
        
    def send(self,sender,reciever,subject,message):
        
        resend.api_key = self.RESEND_API

        try:
            self.response = resend.Emails.send({
            "from": f"{sender}",
            "to": f"{reciever}",
            "subject": f"{subject}",
            "html": f"<strong>{message}</strong>"
            })
        except Exception as e:
            print(f"Error occure while sending the email : {e}")
        
        return self.response


if __name__ == "__main__":
    mail_sender = MailAlert()
    sender = "onboarding@resend.dev"
    reciever = "faizanfarhad2211@gmail.com"
    subject = "mail sender testing"
    message = "this message currently is in testing"
    response = mail_sender.send(sender,reciever,subject,message)
    print(response)