import cv2
from matplotlib import pyplot as plt
import numpy as np

cameraman = cv2.imread("Images/cameraman.tif",cv2.IMREAD_GRAYSCALE)
fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(cameraman, cmap='gray')
ax1.set_title('Original Image')

# a. design a spatial-domain averaging filter whose output is the average of the
# four neighbors of the center pixel in a 3×3 neighborhood.

def four_neighbor_average(image):
    kernel = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float32)/4.
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel ,borderType = cv2.BORDER_DEFAULT)


ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(four_neighbor_average(cameraman), cmap='gray')
ax2.set_title('Spatially filtered image')

#b. Use the freqz2 function, implemented in the lecture, to obtain its frequency-
# domain equivalent and plot the resulting filter function.

def freqz2(fn, N=64):
    """
    Compute the 2D frequency response of a spatial filter.
    Parameters:
    fn: 2D array
    The spatial-domain filter/kernel.
    N: int, optional
    The size of the frequency-domain response (default is 64).
    The filter is zero-padded or truncated to this size.
    Returns:
    f: 1D array
    Frequency coordinates (centered).
    h: 2D array
    2D frequency response of the filter (centered).
    """
    h = np.fft.fftshift(np.fft.fft2(fn, [N, N]))
    f = np.fft.fftshift(np.fft.fftfreq(N))
    return f, h

kernel = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float32)/4.
f, h = freqz2(kernel, N=cameraman.shape[0])

F_x, F_y = np.meshgrid(f, f)
#need to use meshgrid to get proper coordintes for the plot_surface function

ax3 = fig.add_subplot(2,2,3, projection='3d')
ax3.plot_surface(F_x, F_y, np.abs(h),cmap='summer', antialiased=False)
ax3.set_title('Filter Response Magnitude')

# c, plot frquency filtered image alonside the other plots

h_unshifted = np.fft.ifftshift(h)
cameraman_fft = np.fft.fft2(cameraman, cameraman.shape)
cameraman_filtered = np.real(np.fft.ifft2(cameraman_fft*h_unshifted))
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(cameraman_filtered,cmap='gray')
ax4.set_title('Frequency Filtered Image')

plt.show()
