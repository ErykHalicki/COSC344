import numpy as np
import cv2
from matplotlib import pyplot as plt

baboon = cv2.imread('Images/BaboonGray.png', cv2.IMREAD_GRAYSCALE)

def imageScaling4e(f, cx, cy):
    height, width = f.shape
    new_height = int(height * cy)
    new_width = int(width * cx)

    scaling_matrix = np.array([[cx, 0, 0],
                               [0, cy, 0],
                               [0, 0, 1]])

    inverse_matrix = np.linalg.inv(scaling_matrix)

    scaled_image = np.zeros((new_height, new_width), dtype=f.dtype)

    for y_new in range(new_height):
        for x_new in range(new_width):
            coords = np.array([x_new, y_new, 1])
            old_coords = inverse_matrix @ coords
            x_old, y_old = old_coords[0], old_coords[1]

            x0 = int(np.round(x_old))
            y0 = int(np.round(y_old))

            if 0 <= x0 < width and 0 <= y0 < height:
                scaled_image[y_new, x_new] = f[y0, x0]

    return scaled_image

def imageRotate4e(f, theta=0, mode='crop'):
    height, width = f.shape
    theta_rad = -np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                [sin_theta, cos_theta, 0],
                                [0, 0, 1]])

    if mode == 'full':
        corners = np.array([[0, 0, 1],
                           [width - 1, 0, 1],
                           [0, height - 1, 1],
                           [width - 1, height - 1, 1]])

        center_x = (width - 1) / 2
        center_y = (height - 1) / 2

        translate_to_origin = np.array([[1, 0, -center_x],
                                        [0, 1, -center_y],
                                        [0, 0, 1]])

        rotated_corners = []
        for corner in corners:
            translated = translate_to_origin @ corner
            rotated = rotation_matrix @ translated
            rotated_corners.append(rotated[:2])

        rotated_corners = np.array(rotated_corners)
        min_x = np.min(rotated_corners[:, 0])
        max_x = np.max(rotated_corners[:, 0])
        min_y = np.min(rotated_corners[:, 1])
        max_y = np.max(rotated_corners[:, 1])

        new_width = int(np.ceil(max_x - min_x)) + 1
        new_height = int(np.ceil(max_y - min_y)) + 1

        translate_back = np.array([[1, 0, (new_width - 1) / 2],
                                   [0, 1, (new_height - 1) / 2],
                                   [0, 0, 1]])
    else:
        new_width = width
        new_height = height
        center_x = (width - 1) / 2
        center_y = (height - 1) / 2

        translate_to_origin = np.array([[1, 0, -center_x],
                                        [0, 1, -center_y],
                                        [0, 0, 1]])

        translate_back = np.array([[1, 0, center_x],
                                   [0, 1, center_y],
                                   [0, 0, 1]])

    full_transform = translate_back @ rotation_matrix @ translate_to_origin
    inverse_transform = np.linalg.inv(full_transform)

    rotated_image = np.zeros((new_height, new_width), dtype=f.dtype)

    for y_new in range(new_height):
        for x_new in range(new_width):
            coords = np.array([x_new, y_new, 1])
            old_coords = inverse_transform @ coords
            x_old, y_old = old_coords[0], old_coords[1]

            x0 = int(np.round(x_old))
            y0 = int(np.round(y_old))

            if 0 <= x0 < width and 0 <= y0 < height:
                rotated_image[y_new, x_new] = f[y0, x0]

    return rotated_image

rotated_baboon_crop = imageRotate4e(baboon, 45, 'crop')
rotated_baboon_full = imageRotate4e(baboon, 45, 'full')

plt.figure()
plt.imshow(baboon, cmap='gray')
plt.title('Original Baboon')
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(rotated_baboon_crop, cmap='gray')
plt.title('Rotated 45° (crop mode)')
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(rotated_baboon_full, cmap='gray')
plt.title('Rotated 45° (full mode)')
plt.axis('off')
plt.show() 


