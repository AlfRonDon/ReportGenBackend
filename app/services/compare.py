from __future__ import annotations

from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def compare_images(ref_img: Path, test_img: Path, out_diff: Path) -> float:
    A = Image.open(ref_img).convert("RGB")
    B = Image.open(test_img).convert("RGB")
    target = (A.width, A.height)
    B = B.resize(target)
    A_arr = np.array(A).astype("uint8")
    B_arr = np.array(B).astype("uint8")
    A_g = cv2.cvtColor(A_arr, cv2.COLOR_RGB2GRAY)
    B_g = cv2.cvtColor(B_arr, cv2.COLOR_RGB2GRAY)
    score, diff = ssim(A_g, B_g, full=True, data_range=255)
    heat = ((1 - diff) * 255).astype("uint8")
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(A_arr, 0.6, heat_color, 0.4, 0)
    cv2.imwrite(str(out_diff), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return float(score)

