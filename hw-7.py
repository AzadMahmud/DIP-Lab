import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


img = cv2.imread('images/tulip.jpg', 0)
if img is None:
    raise FileNotFoundError("Image not found. Check the path!")


def calculate_histogram(image):
    histogram = [0] * 256
    for pixel in image.flatten():
        histogram[pixel] += 1
    return histogram


def histogram_equalization(img):
    histogram, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    
    pdf = histogram / histogram.sum()
    cdf = np.cumsum(pdf)
    
    
    cdf_normalized = (cdf / cdf.max() * 255).astype(np.uint8)
    
    
    equalized_img = cdf_normalized[img]
    
    return equalized_img, cdf


equalized_img_manual, cdf_manual = histogram_equalization(img)


equalized_img_cv2 = cv2.equalizeHist(img)


plt.figure(figsize=(14, 10))


plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')


original_hist = calculate_histogram(img)
plt.subplot(3, 3, 2)
sns.histplot(x=np.arange(256), weights=original_hist, bins=256, color='blue', alpha=0.7)
plt.title('Original Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')


plt.subplot(3, 3, 3)
plt.plot(np.arange(256), np.cumsum(original_hist) / sum(original_hist), color='red')
plt.title('Original CDF')
plt.xlabel('Pixel Intensity')
plt.ylabel('CDF')
plt.grid(True)


plt.subplot(3, 3, 4)
plt.imshow(equalized_img_manual, cmap='gray')
plt.title('Manual Equalized Image')
plt.axis('off')


manual_hist = calculate_histogram(equalized_img_manual)
plt.subplot(3, 3, 5)
sns.histplot(x=np.arange(256), weights=manual_hist, bins=256, color='green', alpha=0.7)
plt.title('Manual Equalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')


plt.subplot(3, 3, 6)
plt.plot(np.arange(256), cdf_manual, color='red')
plt.title('Manual Equalized CDF')
plt.xlabel('Pixel Intensity')
plt.ylabel('CDF')
plt.grid(True)


plt.subplot(3, 3, 7)
plt.imshow(equalized_img_cv2, cmap='gray')
plt.title('OpenCV Equalized Image')
plt.axis('off')


cv2_hist, _ = np.histogram(equalized_img_cv2.flatten(), bins=256, range=(0,256))
plt.subplot(3, 3, 8)
sns.histplot(x=np.arange(256), weights=cv2_hist, bins=256, color='orange', alpha=0.7)
plt.title('OpenCV Equalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')


plt.subplot(3, 3, 9)
plt.plot(np.arange(256), np.cumsum(cv2_hist) / sum(cv2_hist), color='red')
plt.title('OpenCV Equalized CDF')
plt.xlabel('Pixel Intensity')
plt.ylabel('CDF')
plt.grid(True)

plt.tight_layout()
plt.savefig('histogram_equalization_comparison.png')

