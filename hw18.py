import cv2
import numpy as np
import scipy.fftpack
import pywt
import os
import heapq
from collections import Counter, defaultdict
import math
import matplotlib.pyplot as plt



def calculate_psnr(img1, img2):
    """Calculates the PSNR between two 8-bit grayscale images."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr



class HuffmanNode:
    """Node for the Huffman Tree."""
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        """Allow nodes to be compared in the priority queue."""
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    """Builds a Huffman Tree from a frequency counter."""
    priority_queue = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        
        merged_freq = left.freq + right.freq
        merged_node = HuffmanNode(None, merged_freq)
        merged_node.left = left
        merged_node.right = right
        
        heapq.heappush(priority_queue, merged_node)
    
    return priority_queue[0]

def generate_huffman_codes(node, prefix="", codebook=None):
    """Recursively generates codes from the Huffman Tree."""
    if codebook is None:
        codebook = {}
    
    if node.char is not None:
        codebook[node.char] = prefix
    else:
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)
    
    return codebook

def huffman_compress(image):
    """
    Performs Huffman encoding on an image and returns the compression ratio.
    """
    
    data = image.flatten()
    frequencies = Counter(data)
    
    
    root_node = build_huffman_tree(frequencies)
    codebook = generate_huffman_codes(root_node)
    
    
    original_bits = data.size * 8  
    
    compressed_bits = 0
    for char, freq in frequencies.items():
        compressed_bits += freq * len(codebook[char])
        
    
    
    
    
    
    
    compression_ratio = original_bits / compressed_bits
    
    return compression_ratio, codebook




Q_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def dct_compress(image, q_matrix):
    """
    Compresses an image using DCT, 8x8 blocks, and quantization.
    Returns the list of quantized blocks and the total non-zero count.
    """
    h, w = image.shape
    image_float = image.astype(float) - 128  
    
    quantized_blocks = []
    non_zero_coeffs = 0
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image_float[i:i+8, j:j+8]
            
            
            dct_coeffs = scipy.fftpack.dctn(block, norm='ortho')
            
            
            quantized_block = np.round(dct_coeffs / q_matrix)
            
            quantized_blocks.append(quantized_block)
            non_zero_coeffs += np.count_nonzero(quantized_block)
            
    return quantized_blocks, non_zero_coeffs

def dct_decompress(quantized_blocks, q_matrix, original_shape):
    """
    Decompresses an image from its quantized DCT blocks.
    """
    h, w = original_shape
    reconstructed_image = np.zeros(original_shape, dtype=float)
    
    block_index = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            quantized_block = quantized_blocks[block_index]
            block_index += 1
            
            
            dequantized_coeffs = quantized_block * q_matrix
            
            
            reconstructed_block = scipy.fftpack.idctn(dequantized_coeffs, norm='ortho')
            
            
            reconstructed_block = reconstructed_block + 128
            reconstructed_block = np.clip(reconstructed_block, 0, 255)
            
            reconstructed_image[i:i+8, j:j+8] = reconstructed_block
            
    return reconstructed_image.astype(np.uint8)



def dwt_compress_decompress(image, threshold_percent=90):
    """
    Compresses and decompresses an image using DWT and hard thresholding.
    Returns the reconstructed image and the sparsity ratio.
    """
    
    
    wavelet = 'db4' 
    
    
    
    coeffs = pywt.dwt2(image.astype(float), wavelet)
    LL, (LH, HL, HH) = coeffs
    
    
    detail_coeffs = np.concatenate((LH.ravel(), HL.ravel(), HH.ravel()))
    
    
    abs_details = np.abs(detail_coeffs)
    threshold_value = np.percentile(abs_details, threshold_percent)
    
    
    LH_t = pywt.threshold(LH, threshold_value, mode='hard')
    HL_t = pywt.threshold(HL, threshold_value, mode='hard')
    HH_t = pywt.threshold(HH, threshold_value, mode='hard')
    
    
    total_coeffs = image.size
    
    non_zero_coeffs = np.count_nonzero(LL) + \
                      np.count_nonzero(LH_t) + \
                      np.count_nonzero(HL_t) + \
                      np.count_nonzero(HH_t)
                      
    sparsity_ratio = total_coeffs / non_zero_coeffs
    
    
    coeffs_thresholded = (LL, (LH_t, HL_t, HH_t))
    reconstructed_image = pywt.idwt2(coeffs_thresholded, wavelet)
    
    
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    
    
    h, w = image.shape
    reconstructed_image = reconstructed_image[:h, :w]
    
    return reconstructed_image.astype(np.uint8), sparsity_ratio



def main():
    
    
    image_path = 'images/barbara.jpg'
    
    
    
    DWT_THRESHOLD_PERCENT = 90.0 
    
    

    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
 
        
    
    h, w = image_gray.shape
    h_new = h - (h % 8)
    w_new = w - (w % 8)
    image_gray = image_gray[:h_new, :w_new]
    
    print(f"Loaded and cropped image to shape: {image_gray.shape}\n")
    
    
    cv2.imwrite('original.png', image_gray)

    
    print("--- Running Huffman (Lossless) ---")
    huffman_cr, _ = huffman_compress(image_gray)
    cv2.imwrite('huffman_reconstructed.png', image_gray)
    print(f"Huffman Compression Ratio: {huffman_cr:.2f}")
    print("Saved 'huffman_reconstructed.png'")

    
    print("\n--- Running DCT (Lossy) ---")
    quantized_blocks, dct_non_zero = dct_compress(image_gray, Q_MATRIX)
    dct_reconstructed = dct_decompress(quantized_blocks, Q_MATRIX, image_gray.shape)
    
    dct_psnr = calculate_psnr(image_gray, dct_reconstructed)
    dct_sr = image_gray.size / dct_non_zero
    
    print(f"DCT PSNR: {dct_psnr:.2f} dB")
    print(f"DCT Sparsity Ratio: {dct_sr:.2f}")
    cv2.imwrite('dct_reconstructed.png', dct_reconstructed)
    print("Saved 'dct_reconstructed.png'")

    
    print("\n--- Running DWT (Lossy) ---")
    dwt_reconstructed, dwt_sr = dwt_compress_decompress(image_gray, DWT_THRESHOLD_PERCENT)
    
    dwt_psnr = calculate_psnr(image_gray, dwt_reconstructed)
    
    print(f"DWT PSNR: {dwt_psnr:.2f} dB")
    print(f"DWT Sparsity Ratio: {dwt_sr:.2f}")
    cv2.imwrite('dwt_reconstructed.png', dwt_reconstructed)
    print("Saved 'dwt_reconstructed.png'")
    
    
    print("\nDisplaying results...")
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.imread('huffman_reconstructed.png', 0), cmap='gray')
    plt.title(f'Huffman (Lossless)\nCR: {huffman_cr:.2f}')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(dct_reconstructed, cmap='gray')
    plt.title(f'DCT (Lossy)\nPSNR: {dct_psnr:.2f} dB, SR: {dct_sr:.2f}')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(dwt_reconstructed, cmap='gray')
    plt.title(f'DWT (Lossy)\nPSNR: {dwt_psnr:.2f} dB, SR: {dwt_sr:.2f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()