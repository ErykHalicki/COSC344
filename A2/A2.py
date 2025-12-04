import numpy as np
import cv2
from matplotlib import pyplot as plt

def mask4e(M,N,rUL,cUL,rLR,cLR):
    mask = np.zeros((M,N), dtype=np.uint8)
    if rUL < rLR and cUL < cLR and rUL<M and rLR<M and cUL<N and cLR<N:
        cv2.rectangle(mask, (cUL, rUL),(cLR, rLR), 1, cv2.FILLED)
    return mask

baboon = cv2.imread('Images_A2/BaboonGray.png', cv2.IMREAD_GRAYSCALE)
camera_man = cv2.imread('Images_A2/cameraman.png', cv2.IMREAD_GRAYSCALE)

'''
Arithmetic operations between grayscale images:

(a) [1 point] Write a function 𝑖𝑚𝐴𝑟𝑖𝑡ℎ𝑚𝑒𝑡𝑖𝑐4𝑒(𝑓1, 𝑓2, 𝑜𝑝) for performing operations 𝑜𝑝 on
𝑓1 and 𝑓2. Parameter 𝑜𝑝 is a character string that indicates the following arithmetic
operations between 𝑓1 and 𝑓2: ‘add’ (𝑓1 + 𝑓2), ‘subtract’ (𝑓1 − 𝑓2), ‘multiply’ (𝑓1 ∗
𝑓2), ‘divide’ (𝑓1/𝑓2). These are elementwise operations. Output image 𝑔 should be
floating point. (Hint: Convert the input images to floating point to perform the arithmetic
operations.)
'''
def imArithmetic4e(f1, f2, op):
    f1 = f1.astype(float) / 255.
    f2 = f2.astype(float) / 255.
    if op=='add':
        return cv2.add(f1, f2)
    if op=='subtract':
        return cv2.subtract(f1, f2)
    if op=='multiply':
        return cv2.multiply(f1, f2)
    if op=='divide':
        return cv2.divide(f1, f2)


'''
(b) [1 point] Read the image “BaboonGray.png” and use A1 function 𝑚𝑎𝑠𝑘 and the
‘multiply’ option in 𝑖𝑚𝐴𝑟𝑖𝑡ℎ𝑚𝑒𝑡𝑖𝑐4𝑒 to highlight part of the baboon’s face (shown
below). This is an example of how to define a region of interest (ROI) in an image. (Hint:
Use A1 exercise function of 𝑐𝑢𝑟𝑠𝑜𝑟𝑉𝑎𝑙𝑢𝑒𝑠4𝑒 to get the coordinates needed to define the
mask image.)
'''
M = baboon.shape[0]
N = baboon.shape[1]
center_mask = mask4e(M, N, 30, 80, 460, 460) 
highlighted_baboon = imArithmetic4e(baboon, center_mask, 'multiply')

plt.subplot(1, 2, 1)
plt.imshow(center_mask, cmap='gray')
plt.title('Mask')
plt.subplot(1, 2, 2)
plt.imshow(highlighted_baboon, cmap='gray')
plt.title('Highlighted Baboon')
plt.show()

'''
User-defined ROI

2. Brightness correction:

(a) [1 point] Write a function 𝑏𝑟𝑖𝑔ℎ𝑡𝑛𝑒𝑠𝑠𝐶𝑜𝑟𝑟(𝑓, 𝑝𝑒𝑟𝑐𝑒𝑛𝑡, 𝑜𝑝) to perform brightness
correction on monochrome images. It should take as arguments a monochrome image
𝑓, a number 𝑝𝑒𝑟𝑐𝑒𝑛𝑡 between 0 and 100 (amount of brightness correction, expressed
in percentage terms), and a third parameter 𝑜𝑝 indicating whether the correction is
intended to brighten (‘brighten’) or darken (‘darken’) the image.
'''
def brightnessCorr(f, percent, op):
    f = f.astype(float) / 255.
    brightness_change = percent / 100.
    if op == 'brighten':
        return np.clip(f + brightness_change, 0, 1)
    if op == 'darken':
        return np.clip(f - brightness_change, 0, 1)

'''
(b) [0.5 point] Read the image "cameraman.png" and brighten it by 30%. Show the result
in a single plot.
'''
brightened_cameraman = brightnessCorr(camera_man, 30, 'brighten')
plt.subplot(1, 2, 1)
plt.imshow(camera_man, cmap='gray')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(brightened_cameraman, cmap='gray')
plt.title('Brightened 30%')
plt.show()

'''
3. Insert a small image into a large image:

(a) [1 point] Write a function 𝑖𝑛𝑠𝑒𝑟𝑡(𝑓1, 𝑓2, 𝑥, 𝑦) to insert a small image 𝑓1 into a large
image 𝑓2 at the location of (𝑥, 𝑦). The function should return the new image.
'''
def insert(f1, f2, x, y):
    result = f2.copy()
    h_small, w_small = f1.shape[:2]
    h_large, w_large = f2.shape[:2]

    y_end = min(y + h_small, h_large)#prevents going over the edge fo large image
    x_end = min(x + w_small, w_large)
    h_insert = y_end - y
    w_insert = x_end - x

    if h_insert > 0 and w_insert > 0:
        roi = result[y:y_end, x:x_end]
        small_img = f1[:h_insert, :w_insert]
        result[y:y_end, x:x_end] = cv2.add(roi, small_img)

    return result

'''
(b) [0.5 point] Insert "small_ubc_logo.jpg" into "cameraman.png" at location (10,10)
and show the result.
'''
ubc_logo = cv2.imread('Images_A2/small_ubc_logo.jpg', cv2.IMREAD_GRAYSCALE)
result_image = insert(ubc_logo, camera_man, 10, 10)
plt.subplot(1, 2, 1)
plt.imshow(camera_man, cmap='gray')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(result_image, cmap='gray')
plt.title('With UBC Logo')
plt.show()
