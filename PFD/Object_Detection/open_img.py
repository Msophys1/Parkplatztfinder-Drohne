import cv2
 
# Load an color image in grayscale
img = cv2.imread('C:/Users/widde/Desktop/PFD/Object_Detection/outputs/detection_output0.png')
 
# show image
cv2.imshow('detection',img)
cv2.waitKey(0)
cv2.destroyAllWindows()