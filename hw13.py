import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ---------- Helper Functions ----------

def adjust_contrast(img, level):
    """Adjust contrast: level ∈ ['low', 'normal', 'high']"""
    if level == 'low':
        return cv2.convertScaleAbs(img, alpha=0.5, beta=30)
    elif level == 'high':
        return cv2.convertScaleAbs(img, alpha=1.5, beta=-30)
    else:
        return img

def apply_fft(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
    return dft, dft_shift, magnitude

def make_filters(shape, low=30, high=80):
    rows, cols = shape
    crow, ccol = rows//2, cols//2

    # Low-pass
    low_pass = np.zeros((rows, cols, 2), np.float32)
    cv2.circle(low_pass, (ccol, crow), low, (1,1,0), -1)

    # High-pass
    high_pass = np.ones((rows, cols, 2), np.float32)
    cv2.circle(high_pass, (ccol, crow), high, (0,0,0), -1)

    # Band-pass
    band_pass = np.zeros((rows, cols, 2), np.float32)
    cv2.circle(band_pass, (ccol, crow), high, (1,1,0), -1)
    cv2.circle(band_pass, (ccol, crow), low, (0,0,0), -1)

    return low_pass, high_pass, band_pass

def apply_filter(dft_shift, mask):
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    return img_back

# ---------- Main Processing ----------

os.makedirs("results", exist_ok=True)
img_paths = sorted(glob.glob("./images/*"))[:5]  # take 5 images

for idx, path in enumerate(img_paths):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    contrasts = ["low", "normal", "high"]
    low_pass, high_pass, band_pass = make_filters(img.shape, 30, 80)

    # We’ll now show 5 columns: original, FFT, low-pass, high-pass, band-pass
    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    fig.suptitle(f"Image {idx+1}: Frequency Domain Analysis", fontsize=14)

    for i, c in enumerate(contrasts):
        adj = adjust_contrast(img, c)
        dft, dft_shift, magnitude = apply_fft(adj)

        img_low = apply_filter(dft_shift, low_pass)
        img_high = apply_filter(dft_shift, high_pass)
        img_band = apply_filter(dft_shift, band_pass)

        # Plot all
        axes[i,0].imshow(adj, cmap='gray')
        axes[i,0].set_title(f"{c.capitalize()} Contrast")

        axes[i,1].imshow(magnitude, cmap='gray')
        axes[i,1].set_title("FFT Magnitude")

        axes[i,2].imshow(img_low, cmap='gray')
        axes[i,2].set_title("Low-pass")

        axes[i,3].imshow(img_high, cmap='gray')
        axes[i,3].set_title("High-pass")

        axes[i,4].imshow(img_band, cmap='gray')
        axes[i,4].set_title("Band-pass")

        for j in range(5):
            axes[i,j].axis('off')

    plt.tight_layout()
    plt.savefig(f"results/image_{idx+1}_analysis.png", dpi=300)
    plt.close(fig)

# Combine all 5 into one overview
# Create 3x2 grid for 5 images (6 slots; last one empty)
fig_all, ax_all = plt.subplots(3, 2, figsize=(10, 12))

# Flatten the 2D array of axes for easy 1D indexing
ax_all = ax_all.flatten()

for i in range(5):  # 5 images total
    img_res = plt.imread(f"results/image_{i+1}_analysis.png")
    ax_all[i].imshow(img_res)
    ax_all[i].axis('off')
    ax_all[i].set_title(f"Image {i+1}")

# Hide the last (6th) unused subplot
ax_all[-1].axis('off')

plt.tight_layout()
plt.savefig("results_all.png", dpi=300)
plt.close(fig_all)