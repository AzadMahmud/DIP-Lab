import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os



def adjust_contrast(img, level):
    if level == 'low':
        return cv2.convertScaleAbs(img, alpha=0.6, beta=0)
    elif level == 'high':
        return cv2.convertScaleAbs(img, alpha=1.6, beta=0)
    else:
        return img

def apply_fft(img, return_mag=True):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    if return_mag:
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        return fshift, magnitude
    return fshift

def apply_filter(fshift, H):
    f_ishift = np.fft.ifftshift(fshift * H)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def make_ideal_filters(shape, D0_low=30, D0_high=80):
    P, Q = shape
    U, V = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((U - Q/2)**2 + (V - P/2)**2)
    
    
    H_lp = (D <= D0_high).astype(np.float64)
    
    
    H_hp = (D > D0_low).astype(np.float64)
    
    
    H_bp = ((D > D0_low) & (D <= D0_high)).astype(np.float64)
    
    return H_lp, H_hp, H_bp

def make_butterworth_filters(shape, D0_low=30, D0_high=80, n=2):
    P, Q = shape
    U, V = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((U - Q/2)**2 + (V - P/2)**2)
    D = np.maximum(D, 1e-5)  
    
    
    H_lp = 1 / (1 + (D / D0_high)**(2 * n))
    
    
    H_hp = 1 / (1 + (D0_low / D)**(2 * n))
    
    
    H_bp = H_hp * H_lp
    
    return H_lp, H_hp, H_bp

def make_gaussian_filters(shape, D0_low=30, D0_high=80):
    P, Q = shape
    U, V = np.meshgrid(np.arange(Q), np.arange(P))
    D2 = (U - Q/2)**2 + (V - P/2)**2
    
    
    H_lp = np.exp(-D2 / (2 * D0_high**2))
    
    
    H_hp = 1 - np.exp(-D2 / (2 * D0_low**2))
    
    
    H_bp = H_hp * H_lp
    
    return H_lp, H_hp, H_bp



os.makedirs("results", exist_ok=True)


img_paths = ["images/dog.jpg"] 
if not img_paths:
    raise ValueError("No images found. Place an image in ./images/ or update the path.")

path = img_paths[0]
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"Failed to load image from {path}.")


img = cv2.resize(img, (256, 256))

contrasts = ["low", "normal", "high"]
filter_types = {
    "ideal": make_ideal_filters,
    "butterworth": make_butterworth_filters,
    "gaussian": make_gaussian_filters
}


def generate_analysis_plot(filter_name, filter_func, filename, **kwargs):
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(f"{filter_name.capitalize()} Filters Analysis", fontsize=14)

    for i, c in enumerate(contrasts):
        adj = adjust_contrast(img, c)
        fshift, magnitude = apply_fft(adj)

        H_lp, H_hp, H_bp = filter_func(adj.shape, **kwargs)

        img_lp = apply_filter(fshift, H_lp)
        img_hp = apply_filter(fshift, H_hp)
        img_bp = apply_filter(fshift, H_bp)

        axes[i, 0].imshow(adj, cmap='gray')
        axes[i, 0].set_title(f"{c.capitalize()} Contrast")

        axes[i, 1].imshow(magnitude, cmap='gray')
        axes[i, 1].set_title("FFT Magnitude")

        axes[i, 2].imshow(img_lp, cmap='gray')
        axes[i, 2].set_title("Low-pass")

        axes[i, 3].imshow(img_hp, cmap='gray')
        axes[i, 3].set_title("High-pass")

        axes[i, 4].imshow(img_bp, cmap='gray')
        axes[i, 4].set_title("Band-pass")

        for j in range(5):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


generate_analysis_plot("ideal", make_ideal_filters, "results/ideal_analysis.png", D0_low=30, D0_high=80)
generate_analysis_plot("butterworth", make_butterworth_filters, "results/butter_analysis.png", D0_low=30, D0_high=80, n=2)
generate_analysis_plot("gaussian", make_gaussian_filters, "results/gauss_analysis.png", D0_low=30, D0_high=80)


adj_normal = adjust_contrast(img, 'normal')
fshift_normal, _ = apply_fft(adj_normal)
ns = [1, 2, 4, 8]
fig, axes = plt.subplots(len(ns), 3, figsize=(10, 12))
fig.suptitle("Butterworth Filters with Varying n (Normal Contrast)", fontsize=14)

for i, n in enumerate(ns):
    H_lp, H_hp, H_bp = make_butterworth_filters(img.shape, 30, 80, n)

    img_lp = apply_filter(fshift_normal, H_lp)
    img_hp = apply_filter(fshift_normal, H_hp)
    img_bp = apply_filter(fshift_normal, H_bp)

    axes[i, 0].imshow(img_lp, cmap='gray')
    axes[i, 0].set_title(f"Low-pass (n={n})")

    axes[i, 1].imshow(img_hp, cmap='gray')
    axes[i, 1].set_title(f"High-pass (n={n})")

    axes[i, 2].imshow(img_bp, cmap='gray')
    axes[i, 2].set_title(f"Band-pass (n={n})")

    for j in range(3):
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig("results/butter_vary_n.png", dpi=300)
plt.close(fig)