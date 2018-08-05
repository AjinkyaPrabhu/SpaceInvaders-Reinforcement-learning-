import cv2
import numpy as np
from PIL import ImageGrab


width = 640
height = 480
while True:
    screen =np.array(ImageGrab.grab())

    frame = cv2.cvtColor(screen,cv2.COLOR_RGB2BGR)

    frame = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
    frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    cv2.imshow('Video',frame)
.....

    frame = cv2.resize(frame,(128,128))



    print(frame)



    if cv2.waitKey(1) and 0XFF == ord('q'):
        break

cv2.destroyAllWindows()
