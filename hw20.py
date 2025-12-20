import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import pandas as pd




def calculate_metrics(original, compressed, label):
    
    if original.shape != compressed.shape:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    
    m_val = mse(original, compressed)
    p_val = psnr(original, compressed)
    
    
    if original.ndim == 3:
        
        s_val = ssim(original, compressed, channel_axis=2, data_range=255)
    else:
        s_val = ssim(original, compressed, data_range=255)
        
    return {
        "Method": label,
        "MSE": round(m_val, 2),
        "PSNR": round(p_val, 2),
        "SSIM": round(s_val, 4)
    }








def rle_encode_simulation(img):
    
    flat = img.flatten()
    if len(flat) == 0: return 0
    
    encoded_len = 0
    prev = flat[0]
    count = 1
    
    for pixel in flat[1:]:
        if pixel == prev:
            count += 1
        else:
            
            encoded_len += 2 
            prev = pixel
            count = 1
    encoded_len += 2 
    
    original_size = len(flat)
    compression_ratio = original_size / encoded_len
    return compression_ratio


def compress_8bit_332(img):
    
    
    b, g, r = cv2.split(img)
    
    
    
    
    
    
    r_3 = r & 0xE0
    g_3 = g & 0xE0
    b_2 = b & 0xC0
    
    
    
    
    reconstructed = cv2.merge([b_2, g_3, r_3])
    return reconstructed


def compress_8bit_palette(img, k=256):
    data = img.reshape((-1, 3))
    data = np.float32(data)
    
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    
    
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


def reduce_bit_depth(img, bits):
    
    levels = 2 ** bits
    div = 256 / levels
    
    
    quantized = np.floor(img / div) * div
    
    
    
    norm_factor = 255 / (levels - 1)
    quantized = (quantized / div) * norm_factor
    
    return np.uint8(quantized)


def compress_dct(img_gray, threshold=0.1):
    
    imf = np.float32(img_gray) / 255.0
    
    
    dst = cv2.dct(imf)
    
    
    dst[np.abs(dst) < threshold] = 0
    
    
    idst = cv2.idct(dst)
    
    
    reconstructed = np.clip(idst * 255, 0, 255).astype(np.uint8)
    return reconstructed


def compress_dwt(img_gray, wavelet='haar', threshold=20):
    
    coeffs = pywt.dwt2(img_gray, wavelet)
    LL, (LH, HL, HH) = coeffs
    
    
    
    LH = pywt.threshold(LH, threshold, mode='soft')
    HL = pywt.threshold(HL, threshold, mode='soft')
    HH = pywt.threshold(HH, threshold, mode='soft')
    
    coeffs_rec = LL, (LH, HL, HH)
    
    
    rec = pywt.idwt2(coeffs_rec, wavelet)
    reconstructed = np.clip(rec, 0, 255).astype(np.uint8)
    
    
    reconstructed = reconstructed[:img_gray.shape[0], :img_gray.shape[1]]
    return reconstructed





def main():
    input_dir = 'images'
    output_dir = 'output_compression'
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created '{input_dir}'. Please add images there and run again.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) < 1:
        print("No images found in input_images folder.")
        return

    all_metrics = []

    for idx, filename in enumerate(image_files):
        print(f"Processing {filename}...")
        img_path = os.path.join(input_dir, filename)
        
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w, _ = img_bgr.shape

        
        img_332 = compress_8bit_332(img_bgr)
        all_metrics.append(calculate_metrics(img_bgr, img_332, f"{filename}_8bit_332"))
        cv2.imwrite(f"{output_dir}/{filename}_332.png", img_332)

        
        img_palette = compress_8bit_palette(img_bgr, k=256)
        all_metrics.append(calculate_metrics(img_bgr, img_palette, f"{filename}_8bit_Palette"))
        cv2.imwrite(f"{output_dir}/{filename}_palette.png", img_palette)

        
        for b in [5, 4, 2, 1]:
            img_reduced = reduce_bit_depth(img_bgr, b)
            all_metrics.append(calculate_metrics(img_bgr, img_reduced, f"{filename}_{b}bit"))
            if idx == 0: 
                cv2.imwrite(f"{output_dir}/{filename}_{b}bit.png", img_reduced)

        
        
        img_dct = compress_dct(img_gray, threshold=0.02) 
        all_metrics.append(calculate_metrics(img_gray, img_dct, f"{filename}_DCT"))
        if idx == 0: cv2.imwrite(f"{output_dir}/{filename}_DCT.png", img_dct)

        
        img_dwt = compress_dwt(img_gray, wavelet='haar', threshold=30)
        all_metrics.append(calculate_metrics(img_gray, img_dwt, f"{filename}_DWT"))
        if idx == 0: cv2.imwrite(f"{output_dir}/{filename}_DWT.png", img_dwt)
        
        
        rle_ratio = rle_encode_simulation(img_gray)
        
        print(f"  RLE Compression Ratio for {filename}: {rle_ratio:.2f}")

    
    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(output_dir, "compression_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nProcessing complete. Results saved to '{output_dir}'.")
    
    
    print("\n=== Average Metrics by Method ===")
    
    df['Method_Type'] = df['Method'].apply(lambda x: x.split('_', 1)[1] if '_' in x else x)
    summary = df.groupby('Method_Type')[['MSE', 'PSNR', 'SSIM']].mean()
    print(summary)

if __name__ == "__main__":
    main()                                                         