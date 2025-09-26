import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

baboon = cv2.imread('images/Baboon.png')

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

'''
[1 point] Read the image “Baboon.png”, convert it to a grayscale image and get a
horizontal intensity scan line in the middle of the image, and plot the scan line.
'''
#plt.plot(scanLine4e(cv2.cvtColor(baboon, cv2.COLOR_BGR2GRAY), int(baboon.shape[0]/2), 'row'))
#plt.show()


'''
[1 point] Write a function 𝑚𝑎𝑠𝑘4𝑒(𝑀, 𝑁, 𝑟𝑈𝐿, 𝑐𝑈𝐿, 𝑟𝐿𝑅, 𝑐𝐿𝑅) for creating a binary mask
of size 𝑀 × 𝑁 with 1’s in the rectangular region defined by upper-left row and column
coordinates 𝑟𝑈𝐿, 𝑐𝑈𝐿, and lower-right coordinates 𝑟𝐿𝑅, 𝑐𝐿𝑅, respectively. All
coordinates are inclusive. The rest of the 𝑀 × 𝑁 mask should be 0’s. Your function
should contain a check to make sure the specified region of 1’s does not exceed the
dimensions of the 𝑀 × 𝑁 region.
'''
def mask4e(M,N,rUL,cUL,rLR,cLR):
    mask = np.zeros((M,N), dtype=np.uint8)
    if rUL < rLR and cUL < cLR and rUL<M and rLR<M and cUL<N and cLR<N:
        cv2.rectangle(mask, (rUL, cUL),(rLR, cLR), 1, cv2.FILLED)
    return mask


'''
[1 point] Read the image “BaboonGray.jpg” and generate a square mask whose sides are
one-half the size of the image in both directions. The mask should be centered on the
image.
'''

baboon_gray = cv2.imread('images/BaboonGray.png')
plt.imshow(baboon_gray)
plt.show()
