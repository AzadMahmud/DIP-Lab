import numpy as np
import cv2
import matplotlib.pyplot as plt
import time





def dft1d(x):

    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    
    for k in range(N):
        
        for n in range(N):
            
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X

def fft1d(x):

    N = len(x)
    
    
    if N <= 1:
        return x

    
    if N % 2 != 0:
        raise ValueError(f"FFT input size must be a power of 2, but got {N}")

    
    
    even = fft1d(x[::2])
    odd = fft1d(x[1::2])

    
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N // 2):
        
        t = np.exp(-2j * np.pi * k / N) * odd[k]
        
        
        X[k]           = even[k] + t
        X[k + N // 2] = even[k] - t
        
    return X





def dft2d(image):
    M, N = image.shape
    
    
    temp_rows = np.zeros((M, N), dtype=np.complex128)
    for i in range(M):
        temp_rows[i, :] = dft1d(image[i, :])
    
    
    output = np.zeros((M, N), dtype=np.complex128)
    for j in range(N):
        output[:, j] = dft1d(temp_rows[:, j])
        
    return output

def fft2d(image):
    M, N = image.shape
    
    
    temp_rows = np.zeros((M, N), dtype=np.complex128)
    for i in range(M):
        temp_rows[i, :] = fft1d(image[i, :])
    
    
    output = np.zeros((M, N), dtype=np.complex128)
    for j in range(N):
        output[:, j] = fft1d(temp_rows[:, j])
        
    return output





def visualize_spectrum(F):

    F_shifted = np.fft.fftshift(F)
    
    
    magnitude = np.abs(F_shifted)
    
    
    spectrum = 20 * np.log(magnitude + 1)
    
    
    spectrum_img = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return spectrum_img





if __name__ == "__main__":
    

        
    img = cv2.imread('/home/azad/Work/DIP-Lab/images/cameraman.png', cv2.IMREAD_GRAYSCALE) 



    
    
    print("--- Test 1: Small Image (32x32) ---")
    
    
    img_small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    
    start_dft = time.time()
    F_dft = dft2d(img_small)
    end_dft = time.time()
    print(f"Naive 2D DFT (32x32) took: {end_dft - start_dft:.4f} seconds")

    
    start_fft = time.time()
    F_fft = fft2d(img_small)
    end_fft = time.time()
    print(f"Scratch 2D FFT (32x32) took: {end_fft - start_fft:.4f} seconds")

    
    print(f"Are results close? {np.allclose(F_dft, F_fft)}")
    
    
    spectrum_small_fft = visualize_spectrum(F_fft)
    
    
    
    print("\n--- Test 2: Larger Image (256x256) ---")
    
    
    img_large = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    
    start_fft_large = time.time()
    F_fft_large = fft2d(img_large)
    end_fft_large = time.time()
    print(f"Scratch 2D FFT (256x256) took: {end_fft_large - start_fft_large:.4f} seconds")

    
    start_np = time.time()
    F_np = np.fft.fft2(img_large)
    end_np = time.time()
    print(f"NumPy's 2D FFT (256x256) took: {end_np - start_np:.6f} seconds (Note: NumPy is C-optimized)")

    
    spectrum_large_fft = visualize_spectrum(F_fft_large)
    
    
    print("\n(Skipping 2D DFT on 256x256 image... it would take minutes!)")

    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_small, cmap='gray')
    plt.title("Original Image (32x32)")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(spectrum_small_fft, cmap='gray')
    plt.title("FFT Spectrum (32x32)")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_large, cmap='gray')
    plt.title("Original Image (256x256)")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(spectrum_large_fft, cmap='gray')
    plt.title("FFT Spectrum (256x256)")
    plt.axis('off')

    plt.tight_layout()
    
    plt.savefig("fft_dft_comparison.png", dpi=300)