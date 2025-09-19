import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure


def img_to_float(image):
    return image.astype(np.float32) / 255.0


def img_to_uint8(image):
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def get_cdf(image):
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    pdf = hist / hist.sum() if hist.sum() else 1
    return np.cumsum(pdf)


def builtin_hist_match(src, ref):
    channel_axis = None if src.ndim == 2 else -1
    return exposure.match_histograms(src, ref, channel_axis=channel_axis)


def custom_match_v1(src_cdf, ref_cdf):
    mapping = np.zeros(256, dtype=np.uint8)
    for val in range(256):
        diff = np.abs(ref_cdf - src_cdf[val])
        mapping[val] = np.argmin(diff)
    return mapping


def custom_match_v2(src_cdf, ref_cdf):
    return np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)


def run_case(src_img, ref_img, case_title):
    src_norm, ref_norm = img_to_float(src_img), img_to_float(ref_img)
    builtin_out = builtin_hist_match(src_norm, ref_norm)

    src_disp, ref_disp, builtin_disp = (
        img_to_uint8(src_norm),
        img_to_uint8(ref_norm),
        img_to_uint8(builtin_out),
    )

    src_cdf = get_cdf(src_img)
    ref_cdf = get_cdf(ref_img)
    mapping_v1 = custom_match_v1(src_cdf, ref_cdf)
    custom_out1 = mapping_v1[src_img]

    mapping_v2 = custom_match_v2(src_cdf, ref_cdf)
    custom_out2 = mapping_v2[src_img]

    labels = ["Source", "Reference", "Built-in", "Custom v1", "Custom v2"]
    outputs = [src_disp, ref_disp, builtin_disp, custom_out1, custom_out2]

    # Make a bigger figure
    plt.figure(figsize=(15, 20))
    for i, out in enumerate(outputs):
        # image
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(out, cmap="gray", vmin=0, vmax=255)
        plt.title(labels[i])
        plt.axis("off")

        # histogram
        plt.subplot(5, 2, 2 * i + 2)
        plt.hist(out.ravel(), bins=256, range=(0, 256), color="black")
        plt.title(labels[i] + " Histogram")
        plt.xlim([0, 256])

    plt.suptitle(case_title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"hist_match_{case_title.replace(' ', '_')}.png", bbox_inches="tight")
    plt.close()


def main():
    # Make sure these paths point to actual different images
    low_contrast = "/home/azad/academic/DIP-Lab/images/low_con.jpg"
    high_contrast = "/home/azad/academic/DIP-Lab/images/high_con.jpg"
    normal_contrast = "/home/azad/academic/DIP-Lab/images/medium_con.jpg"

    test_cases = [
        ("Low src vs Normal ref", low_contrast, normal_contrast),
        ("Low src vs High ref", low_contrast, high_contrast),
        ("Low src vs Low ref", low_contrast, low_contrast),

        ("Normal src vs Normal ref", normal_contrast, normal_contrast),
        ("Normal src vs High ref", normal_contrast, high_contrast),
        ("Normal src vs Low ref", normal_contrast, low_contrast),

        ("High src vs Normal ref", high_contrast, normal_contrast),
        ("High src vs High ref", high_contrast, high_contrast),
        ("High src vs Low ref", high_contrast, low_contrast),
    ]
    for title, src_path, ref_path in test_cases:
        src_img = cv2.imread(src_path, 0)
        ref_img = cv2.imread(ref_path, 0)
        if src_img is None or ref_img is None:
            raise FileNotFoundError(f"Check file paths:\n  src: {src_path}\n  ref: {ref_path}")
        run_case(src_img, ref_img, title)


if __name__ == "__main__":
    main()
