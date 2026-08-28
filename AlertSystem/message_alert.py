# from mailersend import MailerSendClient, SmsSendingBuilder

# ms = MailerSendClient()

# try:
#     request = (
#         SmsSendingBuilder()
#         .from_number("+1234567890")
#         .to(["+1098765432"])
#         .text("Hello from MailerSend SMS!")
#         .build()
#     )

#     response = ms.sms_sending.send(request)

#     print("✅ SMS sent successfully")

#     if response:
#         print("📨 Response:", response)

# except Exception as e:
#     print("❌ Failed to send SMS")
#     print(f"   Error: {e}")