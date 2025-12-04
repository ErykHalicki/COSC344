import numpy as np
import cv2
from scipy.ndimage import label
from matplotlib import pyplot as plt

image = cv2.imread("Images/bwimage.png", cv2.IMREAD_GRAYSCALE)

_, image = cv2.threshold(image, 127,255, type=cv2.THRESH_BINARY)

labeled_array, num_features = label(image)

centroid_positions = []
for i in range(1,num_features+1):
    y_coords, x_coords = np.where(labeled_array == i)
    mean_y = np.mean(y_coords)
    mean_x = np.mean(x_coords)

    centroid_positions.append((np.mean(y_coords), np.mean(x_coords)))

labeled_array = 255/num_features * labeled_array
labeled_array = labeled_array.astype(np.uint8)

color_mapped_image = cv2.applyColorMap(labeled_array, cv2.COLORMAP_HSV)
color_mapped_image[np.where(labeled_array == 0)] = np.array([255,255,255])

plt.figure(figsize=(6, 6))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(labeled_array, cmap='gray')
plt.title('Labeled Array')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(color_mapped_image, cv2.COLOR_BGR2RGB))
plt.title('Colored Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(color_mapped_image, cv2.COLOR_BGR2RGB))
for (y, x) in centroid_positions:
    plt.plot(x, y, '+', color='black', markersize=15, markeredgewidth=2)
plt.title('Colored Image with Centroids')
plt.axis('off')

plt.tight_layout()
plt.show()
