import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_filter2D(image, kernel, mode="same"):

    
    kernel = np.flipud(np.fliplr(kernel))
    
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    if mode == "same":
        pad_h, pad_w = k_h // 2, k_w // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    elif mode == "valid":
        padded = image
    else:
        raise ValueError("Mode must be 'same' or 'valid'")
    
    out_h = padded.shape[0] - k_h + 1
    out_w = padded.shape[1] - k_w + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)
    
    
    for i in range(out_h):
        for j in range(out_w):
            region = padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
    
    return output



image = cv2.imread("/home/azad/academic/DIP-Lab/images/tulip.jpg", cv2.IMREAD_GRAYSCALE)  
if image is None:
    raise ValueError("Image not found. Please provide a valid image path.")


kernels = {
    "Average": (1/9) * np.ones((3,3)),
    "Sobel_X": np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
    "Sobel_Y": np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
    "Prewitt_X": np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
    "Prewitt_Y": np.array([[-1,-1,-1],[0,0,0],[1,1,1]]),
    "Scharr_X": np.array([[-3,0,3],[-10,0,10],[-3,0,3]]),
    "Scharr_Y": np.array([[-3,-10,-3],[0,0,0],[3,10,3]]),
    "Laplace": np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]),
    
    "Diagonal_Edge": np.array([[-1,0,1],[0,0,0],[1,0,-1]]),
    "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
    "Box_Edge": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
    "Gaussian_Approx": (1/16) * np.array([[1,2,1],[2,4,2],[1,2,1]])
}


plt.figure(figsize=(12, 10))
plt.subplot(3, 5, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

i = 2
for name, k in kernels.items():
    filtered = custom_filter2D(image, k, mode="same")
    plt.subplot(3, 5, i)
    plt.imshow(filtered, cmap="gray")
    plt.title(name)
    plt.axis("off")
    i += 1

plt.tight_layout()
plt.savefig('spatial_Filtering.png')
# plt.show()
