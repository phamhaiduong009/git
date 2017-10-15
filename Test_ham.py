import cv2
import numpy as np
image =cv2.imread("hinhtron.jpg")
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
lower_color = np.array([0,0,0])
upper_color = np.array([0,70,255])
    
mask = cv2.inRange(hsv, lower_color, upper_color)
cv2.imshow("image",image)
cv2.imshow("Binary_image",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
#green = np.uint8([[[60,20,220 ]]])
#hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
#print hsv_green

#import numpy as np
#import cv2

# cap = cv2.VideoCapture(0)

# Capture video from file
#cap = cv2.VideoCapture('BlueUmbrella.webm')

#while(cap.isOpened()):
    
#    ret, frame = cap.read()

#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#    cv2.imshow('frame',gray)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()

#cv2.destroyAllWindows()
