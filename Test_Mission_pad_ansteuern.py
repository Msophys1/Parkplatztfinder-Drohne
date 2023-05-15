from time import sleep
from djitellopy import Tello
import cv2

tello = Tello()
tello.connect()
print(tello.get_battery())
tello.enable_mission_pads()
# Kamera der Drohne starten
tello.streamon()
frame_read = tello.get_frame_read()

# Losfliegen
tello.takeoff()

# Hochfliegen auf 1500 cm
tello.send_rc_control(0, 0, 80, 0)
height = 0
while height < 2000:  # 1 Meter
     sleep(1)
     height = tello.get_height()
     print(height)
tello.send_rc_control(0, 0, 0, 0)

#Bild machen
cv2.imwrite(f'Images/outside.png', frame_read.frame)

tello.send_rc_control(0, 0, -80, 0)
while height > 100:
    height = tello.get_height()
tello.send_rc_control(0, 0, 0, 0)

#landing
# Set initial distance thresholds
DIST_THRESH_X = 15
DIST_THRESH_Y = 15
error = -100

# Loop until drone lands on mission pad
while True:
    # Get distance to mission pad in x and y directions
    dist_x = tello.get_mission_pad_distance_x()
    dist_y = tello.get_mission_pad_distance_y()
    print(dist_y, dist_x)

    # If distance in either direction is less than 10 cm, land on the mission pad
    if dist_x < DIST_THRESH_X and dist_y < DIST_THRESH_Y and dist_x > -DIST_THRESH_X and dist_y > -DIST_THRESH_Y:
        while True:
            dist_x = tello.get_mission_pad_distance_x()
            dist_y = tello.get_mission_pad_distance_y()
            print(dist_y, dist_x)

            DIST_THRESH_X = 5
            DIST_THRESH_Y = 5
            if dist_x < DIST_THRESH_X and dist_x > -DIST_THRESH_X:
                sleep(1)
                tello.send_rc_control(0, 0, 0, 0)

            if dist_y < DIST_THRESH_Y and dist_y > -DIST_THRESH_Y:
                sleep(1)
                tello.send_rc_control(0, 0, 0, 0)

            # Adjust drone position to steer closer to mission pad
            if dist_x > DIST_THRESH_X:
                tello.send_rc_control(0, -10, 0, 0)
            elif dist_x < -DIST_THRESH_X:
                tello.send_rc_control(0, 10, 0, 0)

            if dist_y > DIST_THRESH_Y:
                tello.send_rc_control(10, 0, 0, 0)
            elif dist_y < -DIST_THRESH_Y:
                tello.send_rc_control(-10, 0, 0, 0)
            sleep(1)

            if dist_x < DIST_THRESH_X and dist_y < DIST_THRESH_Y and dist_x > -DIST_THRESH_X and dist_y > -DIST_THRESH_Y:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                break

    if dist_x <= error and dist_y <= error:
        sleep(1)
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
        break

    if dist_x < DIST_THRESH_X and dist_x > -DIST_THRESH_X:
        sleep(1)
        tello.send_rc_control(0, 0, 0, 0)

    if dist_y < DIST_THRESH_Y and dist_y > -DIST_THRESH_Y:
        sleep(1)
        tello.send_rc_control(0, 0, 0, 0)

    # Adjust drone position to steer closer to mission pad
    if dist_x > DIST_THRESH_X:
        tello.send_rc_control(0,-20,0,0)
    elif dist_x < -DIST_THRESH_X:
        tello.send_rc_control(0,20,0,0)

    if dist_y > DIST_THRESH_Y:
        tello.send_rc_control(20,0,0,0)
    elif dist_y < -DIST_THRESH_Y:
        tello.send_rc_control(-20,0,0,0)

# Disconnect from Tello EDU drone
tello.disconnect()