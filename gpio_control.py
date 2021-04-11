from gpiozero import LED
from time import sleep


motor1 = LED(14)
motor2 = LED(15)
motor3 = LED(18)

motor1.off()
motor2.off()
motor3.off()

sleep(1)

motor1.on()
sleep(1)
motor1.off()
