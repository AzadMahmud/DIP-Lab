import cv2
import numpy as np
import matplotlib.pyplot as plt

def simple_canny(gray, use_gaussian=True, ksize=5, sigma=1.0, low=50, high=150):
  
    
    if use_gaussian:
        
        if ksize % 2 == 0:
            ksize += 1
        
        img = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    else:
        img = gray.copy()

    
    
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
                   
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)

    Gx = cv2.filter2D(img, cv2.CV_32F, Kx)
    Gy = cv2.filter2D(img, cv2.CV_32F, Ky)

    
    mag = np.sqrt(Gx**2 + Gy**2)
    
    mag = mag / (mag.max() + 1e-8) * 255
    
    
    angle = np.arctan2(Gy, Gx) * 180.0 / np.pi
    angle[angle < 0] += 180

    M, N = mag.shape

    
    nms = np.zeros((M, N), dtype=np.float32)

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            
            elif 22.5 <= angle[i, j] < 67.5:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            
            elif 67.5 <= angle[i, j] < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            
            elif 112.5 <= angle[i, j] < 157.5:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            if (mag[i, j] >= q) and (mag[i, j] >= r):
                nms[i, j] = mag[i, j]
            else:
                nms[i, j] = 0

    
    strong = 255
    weak = 50 
    
    result = np.zeros((M, N), dtype=np.uint8)

    strong_i, strong_j = np.where(nms >= high)
    weak_i, weak_j = np.where((nms >= low) & (nms < high))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    
    
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if result[i, j] == weak:
                if (result[i-1:i+2, j-1:j+2] == strong).any():
                    result[i, j] = strong
                else:
                    result[i, j] = 0

    return result

def main():
    
    image_path = "images/cameraman.png" 
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    low_thresh = 30
    high_thresh = 100

    
    
    
    kernel_sizes = [1, 3, 5, 7]
    results_kernels = []

    print("Running Kernel Size Analysis...")
    for k in kernel_sizes:
        edges = simple_canny(img_gray, use_gaussian=True, ksize=k, 
                             low=low_thresh, high=high_thresh)
        results_kernels.append(edges)

    
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(img_gray, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    for i, (k, edges) in enumerate(zip(kernel_sizes, results_kernels)):
        plt.subplot(2, 3, i + 2)
        plt.imshow(edges, cmap="gray")
        plt.title(f"Gaussian Kernel: {k}x{k}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("ced_kernel_comparison.png") 
    plt.show()

    
    
    
    edges_no_gauss = simple_canny(img_gray, use_gaussian=False, 
                                  low=low_thresh, high=high_thresh)
    
    
    edges_with_gauss = simple_canny(img_gray, use_gaussian=True, ksize=5, 
                                    low=low_thresh, high=high_thresh)

    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(edges_no_gauss, cmap="gray")
    plt.title("A. Without Gaussian Smoothing")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(edges_with_gauss, cmap="gray")
    plt.title("B. With Gaussian Smoothing (5x5)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("ced_ablation.png") 

    plt.show()

if __name__ == "__main__":
    main()