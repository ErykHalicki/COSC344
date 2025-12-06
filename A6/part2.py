import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
Modify the Python code (“Threshold.py” in Lecture 17) to perform iterative threshold
selection on an input gray-level image to include a variable that counts the number of
iterations and an array that stores the values of T for each iteration. Display the results in a
single figure
'''

I = cv2.imread("coins.png", cv2.IMREAD_GRAYSCALE)
Id = I.astype(np.float32) / 255.0

T = 0.5 * (Id.min() + Id.max())
deltaT = 0.01
done = False
thresholds = []
thresholds.append(T)

while not done:
    g = Id >= T
    Tnext = 0.5 * (Id[g].mean() + Id[~g].mean())
    done = abs(T - Tnext) < deltaT
    T = Tnext
    thresholds.append(T)

print(f"Converged in {len(thresholds)}, final t = {thresholds[-1]}\n threshold history: {thresholds}")

binary = (Id >= T).astype(np.uint8)

'''
Build upon (4) to fill parts of the missing mask (check “floodfill.py” in Lecture 16) for better
background separation. Display the results in a single figure.

(Includes part 4)
'''
def select_seed_points(binary_image):
    seed_points = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            seed_points.append((x, y))
            plt.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
            plt.draw()
            print(f"Seed point selected: ({x}, {y})")

    fig, ax = plt.subplots()
    ax.imshow(binary_image, cmap='gray')
    ax.set_title("Click to select seed points, then close window")
    ax.axis('off')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    return seed_points


seed_points = select_seed_points(binary)

height, width = binary.shape
mask = np.zeros((height + 2, width + 2), np.uint8)

for seed in seed_points:
    mask[:] = 0
    cv2.floodFill(binary, mask, seedPoint=seed, newVal=1, loDiff=0, upDiff=0)
    # must have difference of 0 when working on the mask, since diff >=1 would fill entire mask with 1s

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(Id, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(binary, cmap="gray")
plt.title(f"Binary Image (T={T:.3f})")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(Id * binary, cmap="gray")
plt.title(f"Background removed")
plt.axis("off")

plt.show()

