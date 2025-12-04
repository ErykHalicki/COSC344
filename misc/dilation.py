import numpy as np
import cv2

a = np.array([[0,0,0,0,0],[0,1,1,0,0], [0,1,1,0,0], [0,0,1,0,0], [0,0,0,0,0]], dtype='uint8')

se = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
c = cv2.dilate(a, se)
print(c)
