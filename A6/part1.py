import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
Write a Python script to generate a test image containing a ramp edge and plot the intensity
profile along a horizontal line of the image and the first and second derivatives of the image.
For the second derivative, first blur the image using a Gaussian filter before computing it.

'''

def generate_ramp_edge(x,y, start_ramp, end_ramp ,start_intensity, end_intensity):
    image = np.zeros((y, x), dtype=np.uint8)

    for col in range(x):
        if col < start_ramp:
            image[:, col] = start_intensity
        elif col > end_ramp:
            image[:, col] = end_intensity
        else:
            ramp_progress = (col - start_ramp) / (end_ramp - start_ramp)
            intensity = start_intensity + ramp_progress * (end_intensity - start_intensity)
            image[:, col] = intensity

    return image

def sobel_filter(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.hypot(sobelx, sobely)
    sobel_mag_norm = sobel_mag / sobel_mag.max()
    sobel_uint8 = (sobel_mag_norm * 255).astype(np.uint8)
    _, I_edge = cv2.threshold(sobel_uint8, int(0.05 * 255), 255, cv2.THRESH_BINARY)
    return I_edge

def gaussian(image):
    result = cv2.GaussianBlur(image, (3, 3), 1.0)
    return result

def laplacian(image):
    laplacian_kernel = np.array([[0,1,0],[1, -4, 1],[0,1,0]], dtype=np.float32)
    result = cv2.filter2D(image.astype(np.float32), -1, laplacian_kernel)
    return np.round(result, 3)

def zero_crossing(img):
    zc = np.zeros(img.shape, dtype=np.uint8)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            patch = img[i - 1 : i + 2, j - 1 : j + 2]
            # changed logic because otherwise i wouldnt get any zero crossing detections
            if (patch.min() < 0 and patch.max() >= 0) or (patch.min() <= 0 and patch.max() > 0):
                zc[i, j] = 255
    return zc



image = generate_ramp_edge(150, 40, 55, 97, 0, 126)

intensity_profile = image[0, :]

sobel_result = sobel_filter(image)
gaussian_result = gaussian(image)
laplacian_result = laplacian(gaussian_result)
zero_crossing_result = zero_crossing(laplacian_result)

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 2)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(intensity_profile)
ax1.set_title('Intensity Profile')
ax1.set_xlabel('Pixel Position')
ax1.set_ylabel('Intensity')
ax1.grid(True)

ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(image, cmap='gray', vmin=0, vmax=255)
ax2.set_title('Ramp Edge Image')

ax3 = fig.add_subplot(gs[1, 1])
ax3.imshow(gaussian_result, cmap='gray', vmin=0, vmax=255)
ax3.set_title('Gaussian Blurred Image')

ax4 = fig.add_subplot(gs[2, 0])
ax4.imshow(sobel_result, cmap='gray', vmin=0, vmax=255)
ax4.set_title('Sobel Filtered Image')

ax5 = fig.add_subplot(gs[2, 1])
ax5.imshow(zero_crossing_result, cmap='gray', vmin=0, vmax=255)
ax5.set_title('Zero Crossing')

plt.tight_layout()
plt.show()

'''
Plot the Laplacian and the intensity profiles to explain the need for blurring the Laplacian
'''

center_row = image.shape[0] // 2
original_profile = image[center_row, :]
blurred_profile = gaussian_result[center_row, :]
laplacian_blurred_profile = laplacian_result[center_row, :]

fig2, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].imshow(laplacian_result, cmap='RdBu', aspect='auto')
axes[0].set_title('Laplacian Function (Full Image)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

axes[1].plot(original_profile, label='Original Image')
axes[1].plot(blurred_profile, label='Gaussian Blurred')
axes[1].plot(laplacian_blurred_profile, label='Laplacian ')
axes[1].set_title('Intensity Profiles Comparison')
axes[1].set_xlabel('Pixel Position')
axes[1].set_ylabel('Intensity')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

