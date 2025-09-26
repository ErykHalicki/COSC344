import cv2

def fry(img):
    img = cv2.multiply(img, 1.2)
    img = cv2.subtract(img, 10)
    return img

def fisheye(img):
    # apply distortion
    return img

def redify(img):
    #make hue slightly redder

def sharpen(img):
    #sharpen image using convolution kernel

def quality_reduction(img):
    #reduce quality using some kind of compression (similar to jpeg 10% quality)

cow_image = cv2.imread("cow.webp")

for i in range(5):
    cow_image = fry(cow_image)

cv2.imshow("locked", cow_image)
cv2.waitKey(-1)
