import cv2
import numpy as np
import matplotlib.pyplot as plt

A = cv2.imread('Images/blob.png', cv2.IMREAD_GRAYSCALE)
_, A = cv2.threshold(A, 127, 255, cv2.THRESH_BINARY)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
B_hat = cv2.flip(B, -1)

erosion_A_B = cv2.erode(A, B)
left_side_eq1 = cv2.bitwise_not(erosion_A_B)

complement_A = cv2.bitwise_not(A)
right_side_eq1 = cv2.dilate(complement_A, B_hat)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(left_side_eq1, cmap='gray')
plt.title('Complement of Eroded')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(right_side_eq1, cmap='gray')
plt.title('Dilation of Complement')
plt.axis('off')

plt.tight_layout()
plt.show()

left_side_eq2 = cv2.dilate(A, B)

erosion_complement_A = cv2.erode(complement_A, B_hat)
right_side_eq2 = cv2.bitwise_not(erosion_complement_A)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(left_side_eq2, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(right_side_eq2, cmap='gray')
plt.title('Complement of Eroded Complement')
plt.axis('off')

plt.tight_layout()
plt.show()

#checking that the images are equal
equivlance_eq_1 = left_side_eq1 == right_side_eq1
equivlance_eq_2 = left_side_eq2 == right_side_eq2

print(f"Eq1 matching entries: {A.shape[0]*A.shape[1]} = {np.count_nonzero(equivlance_eq_1)}")
print(f"Eq2 matching entries: {A.shape[0]*A.shape[1]} = {np.count_nonzero(equivlance_eq_2)}")
