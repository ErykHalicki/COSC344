import numpy as np
import cv2
from matplotlib import pyplot as plt

# Q1 ------

def mask4e(M,N,rUL,cUL,rLR,cLR):
    mask = np.zeros((M,N), dtype=np.uint8)
    if rUL < rLR and cUL < cLR and rUL<M and rLR<M and cUL<N and cLR<N:
        cv2.rectangle(mask, (cUL, rUL),(cLR, rLR), 1, cv2.FILLED)
    return mask


roi_coords = []

def on_click(event):
    global roi_coords
    if event.inaxes and event.button == 1:
        roi_coords.append((int(event.xdata), int(event.ydata)))
        if len(roi_coords) == 2:
            plt.close()

def selectROI(image):
    global roi_coords
    roi_coords = []

    M, N = image.shape

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title('Click two corners of ROI')

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if len(roi_coords) == 2:
        cUL = min(roi_coords[0][0], roi_coords[1][0])
        cLR = max(roi_coords[0][0], roi_coords[1][0])
        rUL = min(roi_coords[0][1], roi_coords[1][1])
        rLR = max(roi_coords[0][1], roi_coords[1][1])
        return M, N, rUL, cUL, rLR, cLR
    return None

baboon = cv2.imread('Images/BaboonGray.png', cv2.IMREAD_GRAYSCALE)

M, N, rUL, cUL, rLR, cLR = selectROI(baboon)
mask = mask4e(M, N, rUL, cUL, rLR, cLR)
roi = baboon[rUL:rLR, cUL:cLR]
equalized_roi = cv2.equalizeHist(roi)
result = baboon.copy()
result[rUL:rLR, cUL:cLR] = equalized_roi

plt.imshow(result, cmap='gray')
plt.title('Histogram Equalized ROI')
plt.show()

# Q2 ------ 

def imageConv(f, h, mode='zero'):
    M, N = f.shape
    h_rows, h_cols = h.shape

    pad_row = h_rows // 2
    pad_col = h_cols // 2

    if mode == 'zero':
        padded = np.pad(f, ((pad_row, pad_row), (pad_col, pad_col)), mode='constant', constant_values=0)
    elif mode == 'replicate':
        padded = np.pad(f, ((pad_row, pad_row), (pad_col, pad_col)), mode='edge')

    h_flipped = np.flip(h)

    result = np.zeros_like(f, dtype=np.float64)

    for i in range(M):
        for j in range(N):
            region = padded[i:i+h_rows, j:j+h_cols]
            result[i, j] = np.sum(region * h_flipped)

    return result


A = np.array([[5, 8, 3, 4, 6, 2, 3, 7],
              [3, 2, 1, 1, 9, 5, 1, 0],
              [0, 9, 5, 3, 0, 4, 8, 3],
              [4, 2, 7, 2, 1, 9, 0, 6],
              [9, 7, 9, 8, 0, 4, 2, 4],
              [5, 2, 1, 8, 4, 1, 0, 9],
              [1, 8, 5, 4, 9, 2, 3, 8],
              [3, 7, 1, 2, 3, 4, 4, 6]])

B = np.array([[2, 1, 0],
              [1, 1, -1],
              [0, -1, -2]])

result_conv = imageConv(A, B)
print(result_conv)
