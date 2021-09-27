import RPI.GPIO as GPIO
import serial 
import time

SERIAL_PORT = "/dev/ttyS0"

ser = serial.Serial(SERIAL_PORT,buadrate=9600,timeout=5)

ser.write("AT+CMGF=1\r")
print("Text mode enabled...")
time.sleep(3)
ser.write('AT+CMGS="PHONE NUMBER"\R')
msg = "test message to rpi..."
print("sending message....")
time.sleep(3)
ser.write(msg+chr(26))
time.sleep(3)
print("message sent....")
