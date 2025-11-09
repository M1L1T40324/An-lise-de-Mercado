import smtplib
from email.mime.text import MIMEText

remetente = "SEU_EMAIL@gmail.com"
senha = "SENHA_DO_APP"  # NÃO é sua senha normal
destinatario = "O_MESMO_EMAIL@gmail.com"

msg = MIMEText("Teste de envio ok ✅")
msg["Subject"] = "Teste"
msg["From"] = remetente
msg["To"] = destinatario

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(remetente, senha)
    server.send_message(msg)

print("Enviado ✅")
