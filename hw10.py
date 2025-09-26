import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "."

def main():
    img_gray = cv2.imread('/home/azad/academic/DIP-Lab/images/Morphological_operations.png', cv2.IMREAD_GRAYSCALE)
    binary_img = np.where(img_gray > 127, 255, 0).astype(np.uint8)

    kernels = {
        'rect': build_kernel('rect'),
        'ellipse': build_kernel('ellipse'),
        'cross': build_kernel('cross'),
        'diamond': build_kernel('diamond')
    }

    ops = ['erosion', 'dilation', 'opening', 'closing', 'tophat', 'blackhat']

    for op in ops:
        results, titles = [img_gray], ['Original']

        for kname, k in kernels.items():
            results.append(apply_builtin(binary_img, k, op))
            titles.append(f"{op} - {kname} (cv2)")

        for kname, k in kernels.items():
            results.append(apply_manual(binary_img, k, op))
            titles.append(f"{op} - {kname} (manual)")

        save_comparison(results, titles, op)

def apply_builtin(img, kernel, operation):
    if operation == 'erosion':
        return cv2.erode(img, kernel, iterations=1)
    elif operation == 'dilation':
        return cv2.dilate(img, kernel, iterations=1)
    elif operation == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif operation == 'tophat':
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    elif operation == 'blackhat':
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

def erosion_custom(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=0)
    out = np.zeros_like(img)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            window = padded[i:i+kh, j:j+kw]
            if np.all(window[kernel == 1] == 255):
                out[i, j] = 255
    return out

def dilation_custom(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=0)
    out = np.zeros_like(img)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            window = padded[i:i+kh, j:j+kw]
            if np.any(window[kernel == 1] == 255):
                out[i, j] = 255
    return out

def opening_custom(img, kernel):
    return dilation_custom(erosion_custom(img, kernel), kernel)

def closing_custom(img, kernel):
    return erosion_custom(dilation_custom(img, kernel), kernel)

def tophat_custom(img, kernel):
    return cv2.subtract(img, opening_custom(img, kernel))

def blackhat_custom(img, kernel):
    return cv2.subtract(closing_custom(img, kernel), img)

def apply_manual(img, kernel, operation):
    if operation == 'erosion':
        return erosion_custom(img, kernel)
    elif operation == 'dilation':
        return dilation_custom(img, kernel)
    elif operation == 'opening':
        return opening_custom(img, kernel)
    elif operation == 'closing':
        return closing_custom(img, kernel)
    elif operation == 'tophat':
        return tophat_custom(img, kernel)
    elif operation == 'blackhat':
        return blackhat_custom(img, kernel)

def build_kernel(kind):
    if kind == 'rect':
        return np.ones((5, 5), np.uint8)
    elif kind == 'ellipse':
        return np.array([[0,1,1,1,0],
                         [1,1,1,1,1],
                         [1,1,1,1,1],
                         [1,1,1,1,1],
                         [0,1,1,1,0]], np.uint8)
    elif kind == 'cross':
        return np.array([[0,0,1,0,0],
                         [0,0,1,0,0],
                         [1,1,1,1,1],
                         [0,0,1,0,0],
                         [0,0,1,0,0]], np.uint8)
    elif kind == 'diamond':
        return np.array([[0,0,1,0,0],
                         [0,1,1,1,0],
                         [1,1,1,1,1],
                         [0,1,1,1,0],
                         [0,0,1,0,0]], np.uint8)
        
def save_comparison(images, titles, op_name):
    n = len(images)
    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(16, 10))
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title, fontsize=9)
        plt.axis('off')

    plt.tight_layout(pad=2.0)
    
    
    save_path = os.path.join(RESULTS_DIR, f"{op_name}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
