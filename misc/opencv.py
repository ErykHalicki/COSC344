import cv2
import numpy as np
import matplotlib.pyplot as plt

cow_image = cv2.imread("cow.webp", cv2.IMREAD_UNCHANGED) # flags for reading different image types

cow_image = cv2.cvtColor(cow_image, cv2.COLOR_BGR2RGB)

print(cow_image.shape)

float_cow = cow_image.astype(np.float32) / 255.0

print(np.max(float_cow))

fig, axs = plt.subplots(1,2, figsize=(10,8))
plt.subplots_adjust(bottom=0.2)
plt.axis('off')

axs[0].imshow(float_cow)

axs[1].imshow(cow_image)

plt.show()

