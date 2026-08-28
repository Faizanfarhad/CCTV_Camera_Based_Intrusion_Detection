"""
Unsupervised video anomaly detection using anomalib's FUVAS model.

Read this before using this module
----------------------------------
FUVAS is an *unsupervised video anomaly detector* (few-shot, PCA-based). It is
NOT a "person entered my ROI" detector. It works like this:

    1. train_model(): fit on a folder of NORMAL-ONLY video clips so the model
       learns what "normal" looks like.
    2. load_and_predict(): load the saved checkpoint and score new clips. A high
       anomaly score means the clip deviates from the normal pattern.

There is no ready-made "CCTV intrusion" checkpoint. The X3D backbone is
pre-trained on Kinetics, but the PCA scoring model must be fitted on your own
normal clips before you can run inference.

anomalib 2.6.0 API notes (why the old code broke):
    * ``VideoDataset`` no longer exists.
    * The built-in video datamodules are ``Avenue``, ``ShanghaiTech`` and
      ``UCSDped``. Each expects a fixed benchmark folder layout, so for custom
      CCTV footage you must arrange your clips in the same layout.
    * ``ShanghaiTech`` (used below) expects:
          root/
          ├── training/videos/01.mp4   # normal clips only
          └── testing/videos/02.mp4    # clips to score (normal + abnormal)

If you only care about "a person entered my ROI", use ``background_model.py``
(YOLOv8 person detection) + the ROI polygon check instead -- that is the faster,
more reliable path for the project described in ``what_to_do.txt``.
"""

from pathlib import Path

from anomalib.data import ShanghaiTech
from anomalib.engine import Engine
from anomalib.models import Fuvas

from DetectionEngine.motion_detection import MotionDetector

MODEL_PATH = "savedModel/fuvas_cctv_model.ckpt"
DATASET_ROOT = "AnomalyDataset"

# Persistent background subtractor shared across frames so the "normal"
# background of the ROI is learned over time (frame-by-frame anomaly scoring).
_motion_detector: MotionDetector | None = None


def detect_anomaly(roi_frame, threshold: float = 0.02) -> tuple[bool, float]:
    """Frame-based anomaly detection on a single ROI crop.

    Uses background subtraction (MOG2): the fraction of the ROI occupied by
    foreground (moving) pixels is treated as the anomaly score. This works
    per-frame with no training and is the practical "is something happening
    inside my restricted zone?" signal.

    Args:
        roi_frame: BGR crop produced by ``extract_roi``.
        threshold: motion fraction above which the ROI is flagged anomalous.

    Returns:
        tuple ``(is_anomaly, score)``.
    """
    global _motion_detector

    if roi_frame is None or roi_frame.size == 0:
        return False, 0.0

    if _motion_detector is None:
        _motion_detector = MotionDetector()

    fg_mask = _motion_detector.apply(roi_frame)
    score = float((fg_mask > 0).sum()) / max(1, fg_mask.size)
    return score >= threshold, score


def train_model(root: str = DATASET_ROOT, checkpoint: str = MODEL_PATH) -> None:
    """Fit FUVAS on normal-only training clips and save the checkpoint.

    Run this once, with normal clips placed at ``root/training/videos/``.
    """
    datamodule = ShanghaiTech(
        root=Path(root),
        clip_length_in_frames=16,
        frames_between_clips=8,
    )
    model = Fuvas(backbone="x3d_s", pre_trained=True)
    engine = Engine()
    engine.fit(datamodule=datamodule, model=model)
    engine.save_checkpoint(model, checkpoint)
    print(f"Model saved to {checkpoint}")


def load_and_predict(
    root: str = DATASET_ROOT,
    checkpoint: str = MODEL_PATH,
) -> list:
    """Load a trained checkpoint and score every clip under ``root``.

    Returns a list of ``Prediction`` objects (one per clip).
    """
    model = Fuvas.load_from_checkpoint(checkpoint)
    datamodule = ShanghaiTech(root=Path(root))
    engine = Engine()
    return engine.predict(datamodule=datamodule, model=model)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
    else:
        results = load_and_predict()
        print(f"Scored {len(results)} clip(s)")
        for i, pred in enumerate(results):
            score = getattr(pred, "pred_score", None)
            label = getattr(pred, "pred_label", None)
            print(f"  [{i}] score={score} label={label}")