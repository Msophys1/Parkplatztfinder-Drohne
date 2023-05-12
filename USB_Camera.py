import time

import cv2
from time import sleep

def main():
    # Try camera indices 1
    cap = cv2.VideoCapture(1)

    # Check if no camera is found
    if not cap.isOpened():
        print("Failed to open any camera")
        return

    # Loop to continuously capture and display video frames
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Check if frame was successfully captured
        if not ret:
            print("Failed to capture frame")
            break

        # Display the captured frame
        cv2.imshow("USB Camera", frame)

        #Bild abspeichern von Drohne
        cv2.imwrite(f'Images/{time.time()}', frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #Für Bilder machen
        sleep(1)

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()