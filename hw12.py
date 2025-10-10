import cv2
import heapq
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def build_huffman_tree(frequencies):

    heap = []
    unique_id = 0
    
    
    for symbol, freq in frequencies.items():
        heapq.heappush(heap, (freq, unique_id, (symbol, None, None)))
        unique_id += 1
    
    
    while len(heap) > 1:
        freq1, _, node1 = heapq.heappop(heap)
        freq2, _, node2 = heapq.heappop(heap)
        merged_freq = freq1 + freq2
        merged_node = (None, node1, node2)  
        heapq.heappush(heap, (merged_freq, unique_id, merged_node))
        unique_id += 1
    
    
    return heap[0][2]


def generate_huffman_codes(root):

    codes = {}
    
    def traverse(node, prefix=""):

        symbol, left, right = node
        
        
        if symbol is not None:
            codes[symbol] = prefix if prefix else "0"
            return
        
        
        if left:
            traverse(left, prefix + "0")
        if right:
            traverse(right, prefix + "1")
    
    traverse(root)
    return codes


def huffman_encode_image(image, top_n=15):

    pixels = image.flatten().tolist()
    total_pixels = len(pixels)
    
    
    frequencies = Counter(pixels)
    
    
    root = build_huffman_tree(frequencies)
    codes = generate_huffman_codes(root)
    
    
    bitstream = ''.join(codes[pixel] for pixel in pixels)
    
    
    original_bits = total_pixels * 8  
    compressed_bits = len(bitstream)
    compression_ratio = round(original_bits / max(1, compressed_bits), 2)
    
    
    print(f"Original bits:      {original_bits:,}")
    print(f"Compressed bits:    {compressed_bits:,}")
    print(f"Compression ratio:  {compression_ratio}×")
    print(f"Space savings:      {round((1 - compressed_bits/original_bits)*100, 2)}%")
    
    
    print(f"\n{'Value':<8} {'Count':<10} {'Probability':<12} {'Code Len':<10} {'Code':<15}")
    print("-" * 65)
    
    sorted_frequencies = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    for pixel_value, count in sorted_frequencies[:top_n]:
        probability = count / total_pixels
        code = codes[pixel_value]
        print(f"{pixel_value:<8} {count:<10} {probability:<12.6f} {len(code):<10} {code:<15}")
    
    return codes


def visualize_histogram(image_jpg, image_png):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    
    axes[0, 0].imshow(image_jpg, cmap='gray')
    axes[0, 0].set_title('JPG Grayscale Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image_png, cmap='gray')
    axes[0, 1].set_title('PNG Grayscale Image')
    axes[0, 1].axis('off')
    
    
    axes[1, 0].hist(image_jpg.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axes[1, 0].set_title('JPG Intensity Histogram')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(image_png.flatten(), bins=256, range=(0, 256), color='green', alpha=0.7)
    axes[1, 1].set_title('PNG Intensity Histogram')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('image_histograms.png')
    # plt.show()


def main():

    jpg_path = "/home/azad/Work/DIP-Lab/images/tulip.jpg"
    png_path = "/home/azad/Work/DIP-Lab/images/tulip.png"
    
    
    img_jpg = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    img_png = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    
    
    if img_jpg is None:
        print(f"Error: Could not load JPG image from {jpg_path}")
        return
    if img_png is None:
        print(f"Error: Could not load PNG image from {png_path}")
        return
    
    print("=" * 70)
    print("HUFFMAN CODING RESULTS FOR JPG IMAGE")
    print("=" * 70)
    codes_jpg = huffman_encode_image(img_jpg, top_n=15)
    
    print("\n" + "=" * 70)
    print("HUFFMAN CODING RESULTS FOR PNG IMAGE")
    print("=" * 70)
    codes_png = huffman_encode_image(img_png, top_n=15)
    
    

    visualize_histogram(img_jpg, img_png)
    
 
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)
    print(f"JPG unique pixel values: {len(codes_jpg)}")
    print(f"PNG unique pixel values: {len(codes_png)}")
    print(f"JPG average code length: {np.mean([len(c) for c in codes_jpg.values()]):.2f} bits")
    print(f"PNG average code length: {np.mean([len(c) for c in codes_png.values()]):.2f} bits")


if __name__ == "__main__":
    main()