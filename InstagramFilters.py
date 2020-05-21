import cv2
import numpy as np
import matplotlib.pyplot as plt

# Blurred Filter
im = cv2.imread('images/house.jpeg')
dst = cv2.GaussianBlur(im, (5, 5), cv2.BORDER_DEFAULT)
plt.imshow(dst)
plt.show()

# Edges Filter
im = cv2.imread('images/house.jpeg')
edges = cv2.Canny(im, 100, 300)
plt.imshow(edges)
plt.show()

# Vintage Filter
im = cv2.imread('images/house.jpeg')
rows, cols = im.shape[:2]
kernel_x = cv2.getGaussianKernel(cols,200)
kernel_y = cv2.getGaussianKernel(rows,200)
kernel = kernel_y * kernel_x.T
filter = 255 * kernel / np.linalg.norm(kernel)
vintage_im = np.copy(im)
for i in range(3):
    vintage_im[:,:,i] = vintage_im[:,:,i] * filter
plt.imshow(vintage_im)
plt.show()