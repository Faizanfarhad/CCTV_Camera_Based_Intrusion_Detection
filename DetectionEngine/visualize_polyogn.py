import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_roi(frame, points):
    """Crop ``frame`` to the bounding rect of the polygon ``points``, keeping
    only pixels that fall inside the polygon.

    Args:
        frame: BGR image (numpy array).
        points: list of ``(x, y)`` polygon vertices.

    Returns:
        tuple ``(roi, roi_mask)``:
            roi      - BGR crop of the polygon's bounding box, with everything
                       outside the polygon blacked out.
            roi_mask - binary mask of the same crop (255 inside polygon).
    """
    frame = frame.copy()
    pts = np.array(points, dtype=np.int32)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    cutout = cv2.bitwise_and(frame, frame, mask=mask)

    x, y, w, h = cv2.boundingRect(pts)
    roi = cutout[y:y + h, x:x + w]
    roi_mask = mask[y:y + h, x:x + w]
    return roi, roi_mask


def show(frame, points):
    """Display the ROI defined by ``points`` using matplotlib."""
    roi, _ = extract_roi(frame, points)

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()