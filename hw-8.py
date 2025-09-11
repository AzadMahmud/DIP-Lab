import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    
    img = cv2.imread("/home/azad/academic/DIP-Lab/images/tulip.jpg",
                     cv2.IMREAD_GRAYSCALE)

    
    blurred = cv2.GaussianBlur(img, (3,3), 0)

    
    edges = cv2.Canny(blurred, 20, 150)



    plt.figure(figsize=(12,6))

    plt.subplot(2,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(blurred, cmap='gray')
    plt.title('After Gaussian Blur')
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    
    plt.subplot(2,3,4)
    plt.hist(img.ravel(), 256, [0,256], color='black')
    plt.title('Histogram - Original')

    plt.subplot(2,3,5)
    plt.hist(blurred.ravel(), 256, [0,256], color='black')
    plt.title('Histogram - Blurred')

    plt.subplot(2,3,6)
    plt.hist(edges.ravel(), 256, [0,256], color='black')
    plt.title('Histogram - Edges')

    plt.tight_layout()
    plt.savefig('canny_edge.png')
    # plt.show()

if __name__ == "__main__":
    main()
