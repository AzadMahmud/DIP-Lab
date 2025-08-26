import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (grayscale for filtering)
img = cv2.imread("images/tulip.jpg", cv2.IMREAD_GRAYSCALE)

# 1. Smoothing / Average Kernel
kernel_avg = np.ones((5, 5), np.float32) / 25
smooth = cv2.filter2D(img, -1, kernel_avg)

# 2. Sobel Kernels (manually defined, but could use cv2.Sobel too)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

sobel_x_img = cv2.filter2D(img, -1, sobel_x)
sobel_y_img = cv2.filter2D(img, -1, sobel_y)

# 3. Prewitt Kernels
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]], dtype=np.float32)

prewitt_x_img = cv2.filter2D(img, -1, prewitt_x)
prewitt_y_img = cv2.filter2D(img, -1, prewitt_y)

# 4. Laplace Kernel
laplace_kernel = np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=np.float32)
laplace_img = cv2.filter2D(img, -1, laplace_kernel)

# Plot all results
titles = ['Original', 'Smoothed',
          'Sobel X', 'Sobel Y',
          'Prewitt X', 'Prewitt Y',
          'Laplace']

images = [img, smooth,
          sobel_x_img, sobel_y_img,
          prewitt_x_img, prewitt_y_img,
          laplace_img]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('kernels.png')
