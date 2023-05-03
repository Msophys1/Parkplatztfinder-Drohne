import cv2
from djitellopy import Tello
import time
global img

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

tello.takeoff()
cv2.imwrite("picture.png", frame_read.frame)
time.sleep(0.3) #falls kombiniert mit Keyboard use

tello.streamoff()
tello.land()