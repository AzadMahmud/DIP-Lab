import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

img = cv2.imread('/images/tulip.jpg', 0)

plt.figure(figsize=(10, 8))
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

def calculate_histogram(image):
    histogram = [0] * 256
    for pixel in image.flatten():
        histogram[pixel] += 1
    return histogram

original_hist = calculate_histogram(img)
plt.subplot(2, 3, 2)
sns.histplot(np.arange(256), weights=original_hist, bins=256, color='blue', alpha=0.7)
plt.title('Original Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

def histogram_equalization(img):
    histogram, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    
    pdf = histogram / histogram.sum()
    
    cdf = np.cumsum(pdf)
    
    cdf_normalized = (cdf * 255).astype(np.uint8)
  
    equalized_img = cdf_normalized[img]
    
    return equalized_img, cdf

equalized_img, cdf = histogram_equalization(img)


plt.subplot(2, 3, 4)
plt.imshow(equalized_img, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

final_hist = calculate_histogram(equalized_img)
plt.subplot(2, 3, 5)
sns.histplot(np.arange(256), weights=final_hist, bins=256, color='green', alpha=0.7)
plt.title('Equalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(2, 3, 6)
plt.plot(cdf, color='red')
plt.title('Cumulative Distribution Function (CDF)')
plt.xlabel('Pixel Intensity')
plt.ylabel('CDF Value')
plt.grid(True)

plt.tight_layout()
plt.show()
