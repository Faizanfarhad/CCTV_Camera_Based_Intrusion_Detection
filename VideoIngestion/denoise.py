import cv2


def denoise_frame(frame, method="fast", strength=10):
    """Denoise a single BGR frame.

    Args:
        frame: BGR image (numpy array).
        method: one of "fast", "nlm", "bilateral", "median", "gaussian", "none".
                **fast** :  It averages similar patches to reduce grain while preserving edges and textures ideal for improving low-light or noisy photos.
                
        strength: filter strength (scaled per method).

    Returns:
        Denoised BGR frame.
    """
    if method == "none" or frame is None:
        return frame

    if method == "fast":
        # Non-local means on a downscaled frame -> much faster, near same quality.
        h, w = frame.shape[:2]
        scale = 0.5
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        small = cv2.fastNlMeansDenoisingColored(
            small, None, strength, strength, 7, 21
        )
        return cv2.resize(small, (w, h))

    if method == "nlm":
        # Full-resolution non-local means (slowest, highest quality).
        return cv2.fastNlMeansDenoisingColored(
            frame, None, strength, strength, 7, 21
        )

    if method == "bilateral":
        # Edge-preserving, faster than NLM.
        return cv2.bilateralFilter(frame, 9, strength * 5, strength * 5)

    if method == "median":
        # Best for salt-and-pepper noise.
        k = 5 if strength > 5 else 3
        return cv2.medianBlur(frame, k)

    if method == "gaussian":
        # Fastest, but softens edges.
        return cv2.GaussianBlur(frame, (5, 5), 0)

    raise ValueError(f"Unknown denoise method: {method}")


def denoise_color_img(img):
    """Backwards-compatible wrapper around the default denoiser."""
    return denoise_frame(img, method="fast")


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    img = cv2.imread("testVideo/download.jpeg")
    if img is None:
        print("Could not read test image. Skipping demo.")
        exit()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(denoise_frame(img, method="fast"), cv2.COLOR_BGR2RGB)

    plt.subplot(121), plt.imshow(img), plt.title("Original")
    plt.subplot(122), plt.imshow(dst), plt.title("Denoised")
    plt.show()
