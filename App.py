from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
import numpy as np
from PIL import Image
from keras import models
import time
from djitellopy import Tello
import cv2


# Define PID [Proportional, Integral, Derivative]
pid = [0.4, 0.4, 0]
# Define table for the update of the previous error [L/R, F/B, U/D, y]
pError = [0, 0, 0, 0]
# Define frame size
wTot, hTot = [224, 224]
# Define reference area for the object
refArea = 10000


def findLanding(video, w, h):
    # load trained model
    model = models.load_model('my_model.h5')
    # convert the frame to grayscale
    im = Image.fromarray(video, 'RGB')
    im = im.resize((224, 224))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)

    # start image recognition
    platform = model.predict(img_array)

    # define the table of value
    center_platform = []
    area_platform = []
    probabilities = []

    for (x, y, pw, ph) in platform:
        # extract the probability of the detected object being the platform
        _, prob = platform.predict_proba(img_array[y:y + ph, x:x + pw])
        probabilities.append(prob[0])
        # draw a rectangle around the detected object
        cv2.rectangle(video, (x, y), (x + pw, y + ph), (0, 0, 255), 2)
        # calculate the center coordinate and the area of the rectangle
        cx = x + pw // 2
        cy = y + ph // 2
        area = pw * ph
        # Implement the coordinates in the tables
        center_platform.append([cx, cy])
        area_platform.append(area)
        # mark the center with a green circle
        cv2.circle(video, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    # sort the detected objects based on their probabilities
    sorted_platform = [x for _, x in sorted(zip(probabilities, platform), reverse=True)]

    # return the frame with the marked detected object with the highest probability
    if len(sorted_platform) != 0:
        spx, spy, spw, sph = sorted_platform[0]
        # calculate the center coordinate and the area of the rectangle
        spcx = spx + spw // 2
        spcy = spy + sph // 2
        sparea = spw * sph
        return video, [[spcx, spcy], [spw, sph], sparea]
    # if nothing is recognized return 0
    else:
        return video, [[0, 0], [w, h], 0]


# This function calculates and sends the corrections to be brought to the engines to land precisely on the landing
# platform. It uses the coordinates of the previous function to align the center of the platform with the center of
# the image of the drone taking into account the external conditions (PID).
def track_platform(drone_video, coordinates_platform, w, h, pid_, previous_error):
    # convert tuple to list
    previous_error = list(previous_error)
    # Get object detection info
    area_platform = coordinates_platform[2]
    center_platform = coordinates_platform[0]

    # Calculate center coordinates of the detected object
    cx = center_platform[0]
    cy = center_platform[1]

    # Calculate center coordinates of the total image
    cx_tot = w / 2
    cy_tot = h / 2

    # left/right adjustments
    # calculate error for the x axe
    lr_error = cx_tot - cx
    # if object is on the left set positive speed calculated with PID and update error
    if lr_error > 0:
        lr_speed = int(np.clip(pid_[0] * lr_error + pid_[1] * (lr_error - previous_error[0]), 0, 100))
        previous_error[0] += int(lr_error)
    # if object is on the right set negative speed calculated with PID and update error
    elif lr_error < 0:
        lr_speed = int(np.clip(pid_[0] * lr_error + pid_[1] * (lr_error - previous_error[0]), -100, 0))
        previous_error[0] += int(lr_error)
    # else don't move
    else:
        lr_speed = 0
        previous_error[0] += int(lr_error)

    # calculate error for the y axe
    fb_error = cy_tot - cy
    # if object is down, set positive speed calculated with PID and update error
    if fb_error > 0:
        fb_speed = int(np.clip(pid_[0] * fb_error + pid_[1] * (fb_error - previous_error[1]), 0, 100))
        previous_error[1] += int(fb_error)
    # if object is up, set negative speed calculated with PID and update error
    elif fb_error < 0:
        fb_speed = int(np.clip(pid_[0] * fb_error + pid_[1] * (fb_error - previous_error[1]), -100, 0))
        previous_error[1] += int(fb_error)
    # Else don't move
    else:
        fb_speed = 0
        previous_error[1] += int(fb_error)

    # calculate the Distance between the drone and the platform
    ud_error = refArea - area_platform
    # if the area of the detected object is smaller than the reference area, calculate the descent of the drone with
    # the PID and update the error
    if ud_error > 0:
        ud_speed = int(np.clip(pid_[0] * ud_error + pid_[1] * (ud_error - previous_error[2]), -20, 20))
        previous_error[2] += int(ud_error)
    # Else stay set speed to 0 and initiate the landing
    else:
        ud_speed = 0
        previous_error[2] += int(ud_error)
        drone_video.land()

    # Send correction to the drone
    drone_video.send_rc_control(lr_speed, fb_speed, ud_speed, 0)

    # Return errors
    return lr_error, fb_error, ud_error, 0


# cap = cv2.VideoCapture(0)
def tello_go(self):
    # define object Tello for connection with drone
    tello = Tello()
    # Establish connection with Tello
    tello.connect()
    # Turn the video stream on
    tello.streamon()
    frame_read = tello.get_frame_read()
    # Print the battery level
    print(tello.get_battery())
    time.sleep(10)
    # drone as to take off
    tello.takeoff()
    # Hochfliegen auf 1500 cm
    tello.send_rc_control(0, 0, 80, 0)
    height = 0
    while height < 2000:  # 1 Meter
        time.sleep(1)
        height = tello.get_height()
        print(height)
    tello.send_rc_control(0, 0, 0, 0)

    # Bild machen
    cv2.imwrite(f'Images/outside.png', frame_read.frame)

    # Loop until drone lands on mission pad
    while True:
        # _, img = cap.read()
        # get the frame from the drone
        img = tello.get_frame_read().frame
        # resize the frame to a normalized sized frame
        img = cv2.resize(img, (wTot, hTot))
        # find the platform with the trained model
        img, info = findLanding(img, wTot, hTot)
        # if nothing's found move a bit and try again
        while info == [[0, 0], [wTot, hTot], 0]:
            tello.send_rc_control(25, 25, 0, 0)
            img, info = findLanding(img, wTot, hTot)
            tello.send_rc_control(-50, -50, 0, 0)
            img, info = findLanding(img, wTot, hTot)
            tello.send_rc_control(0, 50, 0, 0)
            img, info = findLanding(img, wTot, hTot)
        # initiate the positioning of the drone over the landing platform and the landing
        pError = track_platform(img, info, wTot, hTot, pid, pError)
        # Show frame
        cv2.imshow("Output", img)
        # security stop/landing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tello.land()
            break
    # Disconnect from Tello EDU drone
    tello.disconnect()


class MainApp(App):
    def build(self):
        #App bauen, mit Text und Druckknopf
        self.icon = "Logo_Drone512.png"
        self.title = "Parkplatzfinder"

        main_layout = BoxLayout(orientation="vertical")

        self.label = Label(
            text='Wilkommen zur Parkplatzfinder Drohne\n    -Stellen sie sicher, dass Sie mit der Tello verbunden sind'
                 '\n    -Stecken Sie die zwei USB-Kabel im Laptop ein'
                 '\n    -Drücken Sie den Knopf der Fernbedienung um die Landestation zu öffnen'
                 '\n    -Halten Sie die Drohne aus dem Fenster und starten Sie das Programm')
        main_layout.add_widget(self.label)

        self.button = Button(
            text="Start", font_size=30, background_color="grey", size_hint = (0.9,0.5),
            pos_hint={"center_x": 0.5, "center_y": 0.8},
        )
        self.button.bind(on_press=self.on_button)
        main_layout.add_widget(self.button)

        return main_layout

    def on_button(self, instance):
        #Bei drücken des Knopfs wird der Text aktuallisiert und der Knopf entfernt
        self.label.text = 'Bitte warten, dies könnte einen Moment dauern' \
                          '\nBehalten Sie die Drohne im Auge'
        self.root.clear_widgets()
        self.root.add_widget(self.label)

        #Das Drohnenprogramm wird gestartet
        Clock.schedule_once(lambda dt: self.tello_go(), 1)

        #Parkplatzanalyse
        #Bild anzeigen
        img = Image(source=f'Parkplatz.png', allow_stretch=True, keep_ratio=False)
        self.root.clear_widgets()
        self.root.add_widget(img)



if __name__ == "__main__":
    app = MainApp()
    app.run()


