# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:39:49 2021

@author: Shashank
"""

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)


GPIO.setup(11,GPIO.OUT)
servo1 = GPIO.PWM(11,50) 

def run_motor():
    servo1.start(0)
    time.sleep(2)
    
    servo1.ChangeDutyCycle(7)
    time.sleep(0.5)
    servo1.ChangeDutyCycle(0)
    time.sleep(5)

    servo1.ChangeDutyCycle(2)
    time.sleep(0.5)
    servo1.ChangeDutyCycle(0)
    
    servo1.stop()
    GPIO.cleanup()