import cv2
import numpy as np
import scipy.fftpack
import pywt
import matplotlib.pyplot as plt

def apply_dft(image):
    """Applies DFT, shifts it, and returns the log-magnitude spectrum."""
    
    f = np.fft.fft2(image)
    
    
    fshift = np.fft.fftshift(f)
    
    
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1) 
    return magnitude_spectrum

def apply_dct(image):
    """Applies 2D DCT and returns the log-magnitude coefficients."""
    
    dct_coeffs = scipy.fftpack.dctn(image, type=2, norm='ortho')
    
    
    log_dct = 20 * np.log(np.abs(dct_coeffs) + 1) 
    return log_dct

def apply_dwt(image):
    """Applies a 1-level DWT and tiles the results into a single image."""
    
    coeffs = pywt.dwt2(image, 'haar')
    
    
    LL, (LH, HL, HH) = coeffs
    
    
    h, w = LL.shape
    
    
    dwt_image = np.zeros((h * 2, w * 2), dtype=image.dtype)
    
    
    dwt_image[0:h, 0:w] = normalize_for_display(LL)
    dwt_image[0:h, w:w*2] = normalize_for_display(LH)
    dwt_image[h:h*2, 0:w] = normalize_for_display(HL)
    dwt_image[h:h*2, w:w*2] = normalize_for_display(HH)
    
    return dwt_image

def normalize_for_display(image_data):
    """Normalizes an array to the 0-255 range for image saving."""
    
    img_norm = image_data - np.min(image_data)
    
    img_norm = img_norm / np.max(img_norm) * 255
    return img_norm.astype(np.uint8)

def main():
    
    
    image_path = 'images/lena.png' 
    
 
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   
        
    
    h, w = image.shape
    if h % 2 != 0:
        image = image[:h-1, :]
    if w % 2 != 0:
        image = image[:, :w-1]
        
    print(f"Loaded image, new shape: {image.shape}")

    
    dft_result = apply_dft(image.astype(float))
    dct_result = apply_dct(image.astype(float))
    dwt_result = apply_dwt(image.astype(float))
    
    
    
    
    
    cv2.imwrite('original_image.png', image)
    
    
    cv2.imwrite('dft_spectrum.png', normalize_for_display(dft_result))
    
    
    cv2.imwrite('dct_coeffs.png', normalize_for_display(dct_result))
    
    
    cv2.imwrite('dwt_coeffs.png', dwt_result) 
    
    print("Successfully processed and saved all images:")
    print("- original_image.png")
    print("- dft_spectrum.png")
    print("- dct_coeffs.png")
    print("- dwt_coeffs.png")

    
    plt.figure(figsize=(12, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(dft_result, cmap='gray')
    plt.title('DFT Log-Magnitude Spectrum')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(dct_result, cmap='gray')
    plt.title('DCT Log-Magnitude Coefficients')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(dwt_result, cmap='gray')
    plt.title('DWT (Haar) 1-Level')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()