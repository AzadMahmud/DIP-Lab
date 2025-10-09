import cv2
import numpy as np
import matplotlib.pyplot as plt


def divide_image(img, s):

    h, w = img.shape
    patch_h = h // s
    patch_w = w // s
    
    tiles = []
    for i in range(s):
        for j in range(s):
            tile = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            tiles.append(tile)
    
    return tiles, patch_h, patch_w


def combine_image(tiles, s, patch_h, patch_w):

    rows = []
    for i in range(s):
        row_tiles = tiles[i*s:(i+1)*s]
        row = np.hstack(row_tiles)
        rows.append(row)
    
    combined = np.vstack(rows)
    return combined


def linear_op1(img):

    result = np.clip(1.3 * img + 25, 0, 255).astype(np.uint8)
    return result


def linear_op2(img):

    result = np.clip(1.1 * img - 35, 0, 255).astype(np.uint8)
    return result


def gamma_correction1(img, gamma=0.7):

    normalized = img / 255.0
    corrected = np.power(normalized, gamma)
    result = np.clip(corrected * 255, 0, 255).astype(np.uint8)
    return result


def gamma_correction2(img, gamma=1.8):

    normalized = img / 255.0
    corrected = np.power(normalized, gamma)
    result = np.clip(corrected * 255, 0, 255).astype(np.uint8)
    return result


def process_image(image_path, grid_size=4):
 
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    plt.figure(figsize=(8, 6))
    plt.imshow(gray_img, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('original.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    
    tiles, patch_h, patch_w = divide_image(gray_img, grid_size)
    
    
    operations = [linear_op1, linear_op2, gamma_correction1, gamma_correction2]
    
    
    processed_tiles = []
    for idx, tile in enumerate(tiles):
        op = operations[idx % len(operations)]
        processed_tile = op(tile.copy())
        processed_tiles.append(processed_tile)
    
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    op_names = ['Linear 1\n(1.3I+25)', 'Linear 2\n(1.1I-35)', 
                'Gamma 1\n(γ=0.7)', 'Gamma 2\n(γ=1.8)']
    
    for idx, (ax, tile) in enumerate(zip(axes.flat, processed_tiles)):
        ax.imshow(tile, cmap='gray')
        ax.set_title(op_names[idx % len(op_names)], fontsize=8)
        ax.axis('off')
    
    plt.suptitle('Image Divided into 4×4 Grid with Different Operations', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig('divided.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    
    reconstructed = combine_image(processed_tiles, grid_size, patch_h, patch_w)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(reconstructed, cmap='gray')
    plt.title('Reconstructed Image with Combined Operations')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('reconstructed.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    
    apply_histogram_methods(reconstructed)
    
    print("Processing complete! Check the output images.")
    return reconstructed

def apply_histogram_methods(img):

    
    he_img = cv2.equalizeHist(img)
    
    
    
    clahe_ahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    ahe_img = clahe_ahe.apply(img)
    
    
    clahe_2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img_2 = clahe_2.apply(img)
    
    clahe_4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_img_4 = clahe_4.apply(img)
    
    clahe_6 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    clahe_img_6 = clahe_6.apply(img)
    
    
    
    h, w = img.shape
    downscaled = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
    
    ahe_bilinear_img = clahe_ahe.apply(upscaled)
    
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    images = [
        (img, 'Original'),
        (he_img, 'Histogram Equalization\n(HE)'),
        (ahe_img, 'Adaptive HE\n(AHE)'),
        (clahe_img_2, 'CLAHE\n(clip=2.0)'),
        (clahe_img_4, 'CLAHE\n(clip=4.0)'),
        (clahe_img_6, 'CLAHE\n(clip=6.0)'),
        (ahe_bilinear_img, 'AHE with\nBilinear Interpolation'),
        (img, 'Original (Reference)')
    ]
    
    for ax, (image, title) in zip(axes.flat, images):
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Comparison of Histogram-Based Enhancement Methods', fontsize=14)
    plt.tight_layout()
    plt.savefig('histogram_methods.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for ax, (image, title) in zip(axes.flat, images):
        ax.hist(image.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        ax.set_title(f'Histogram: {title}', fontsize=9)
        ax.set_xlim([0, 256])
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
    
    plt.suptitle('Histograms of Different Enhancement Methods', fontsize=14)
    plt.tight_layout()
    plt.savefig('histogram_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    
    image_path ="/home/azad/Work/DIP-Lab/images/banksy.jpg"  
    
    
    processed_img = process_image(image_path, grid_size=4)
    