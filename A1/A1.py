import cv2
import numpy as np
import math

'''
1 point] Write a function, 𝑠𝑐𝑎𝑛𝐿𝑖𝑛𝑒4𝑒(𝑓, 𝑙, 𝑙𝑜𝑐), where 𝑓 is a grayscale image, 𝑙 is an
integer, and 𝑙𝑜𝑐 is a character string with value ‘row’ or ‘col’ indicating that 𝑙 is either a
row or column number in image 𝑓. The output, 𝑠, is a vector containing the pixel values
along the specified row or column in 𝑓.
'''
def scanLine4e(f,l,loc):
    if loc=='col':
        return f[: , l]
    if loc=='row':
        return f[l, : ]

arr = np.array([[0,0,1,0,0,0],
                               [1,0,1,0,0,0],
                               [0,1,0,0,0,0],
                               [0,0,1,0,0,1],
                               [1,1,0,0,0,1],
                               [1,1,1,1,0,0]])

print(scanLine4e(arr,3,'row'))

