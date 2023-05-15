from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from time import sleep
from djitellopy import Tello
import cv2
from threading import Thread


def tello_go(self):
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

    # Bild machen
    cv2.imwrite(f'Images/outside.png', frame_read.frame)

    tello.send_rc_control(0, 0, -80, 0)
    while height > 100:
        height = tello.get_height()
    tello.send_rc_control(0, 0, 0, 0)

    # landing
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
            tello.send_rc_control(0, -20, 0, 0)
        elif dist_x < -DIST_THRESH_X:
            tello.send_rc_control(0, 20, 0, 0)

        if dist_y > DIST_THRESH_Y:
            tello.send_rc_control(20, 0, 0, 0)
        elif dist_y < -DIST_THRESH_Y:
            tello.send_rc_control(-20, 0, 0, 0)

    # Disconnect from Tello EDU drone
    tello.disconnect()


def camera_go():
    # Auf USB Kamera zugreiffen
    cap = cv2.VideoCapture(1)

    while True:
        # Livestream anzeigen
        ret, frame = cap.read()
        cv2.imshow('Live Stream', frame)

        # exit if the user presses 'q'
        if not thread_drohne.is_alive():
            cap.release()
            cv2.destroyAllWindows()
            break


thread_drohne = Thread(target=tello_go)
thread_camera = Thread(target=camera_go)


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
        Clock.schedule_once(lambda dt: self.start_tello(), 1)

    def start_tello(self):
        thread_drohne.start()
        thread_camera.start()

#Parkplatzanalyse


        #Bild anzeigen
        img = Image(source=f'images/parkplatz.png', allow_stretch=True, keep_ratio=False)
        self.root.clear_widgets()
        self.root.add_widget(img)



if __name__ == "__main__":
    app = MainApp()
    app.run()


