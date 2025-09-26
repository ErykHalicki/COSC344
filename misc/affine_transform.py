import cv2
import numpy as np
import math
#generate affine transofmation matridx for:
#rotation clockwise by 30 degrees

theta = math.radians(-30)
rotation_matrix_temp = [[math.cos(theta), math.sin(theta),0],
                        [math.sin(theta), math.cos(theta), 0],
                        [0,0,1]]

rotation_matrix = np.array(rotation_matrix_temp, dtype = float)

print(rotation_matrix)

scaling_matrix_temp = [[3.5, 0,0],
                        [0, 3.5, 0],
                        [0,0,1]]
