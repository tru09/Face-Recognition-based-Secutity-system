import time
import serial

def r():
    data = 0
    phone = serial.Serial("/dev/ttyS0", 9600, timeout=5)
    phone.write(b'ATZ\r')
    # print(phone.readall())

    phone.write(b"AT+CMGF=1\r")
    # print(phone.readall())

    # phone.write(b'AT+CMGL="ALL"\r')
    phone.write(b'AT+CMGR=1\r')
    data = phone.readall()
    phone.write(b'AT+CMGD=1\r')
    print(data)
    time.sleep(5)
    if data:
        return data
    return data
