import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


def SendMail(ImgFileName):
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Alert'
    msg['From'] = 'projectfinal009@gmail.com'
    msg['To'] = 'sajgaonkar007@gmail.com'

    text = MIMEText("test")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)

    s = smtplib.SMTP('smtp.gmail.com', 25)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login('projectfinal009@gmail.com', 'Projectfinal@009')
    s.sendmail('projectfinal009@gmail.com', 'sajgaonkar007@gmail.com', msg.as_string())
    s.quit()
