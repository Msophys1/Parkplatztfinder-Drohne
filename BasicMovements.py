from djitellopy import tello
from time import sleep, time
import cv2


me = tello.Tello()
me.connect()

battery: int = me.get_battery()
print(battery)

me.takeoff() # décole
me.send_rc_control(0,30,0,0) # avance
sleep(2)
me.send_rc_control(0,0,0,100) # 180
sleep(2)
me.send_rc_control(0,30,0,0) # avance
sleep(2)
me.send_rc_control(0,0,0,100) # 180
sleep(2)
me.send_rc_control(0,0,0,0) # arrêt moteur
me.land() # attéri
me.takeoff()
img = me.get_frame_read().frame
cv2.imshow('Image', img)
cv2.waitKey(5)
sleep(2)
me.land()






