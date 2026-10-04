"""depth anything v2 wrapper -> inverse depth in [0,1], higher = closer.

verified empirically on code/depthbench/assets/nyc.png (street photo):
raw predicted_depth is already disparity-like inverse depth -- far sky
pixels -> 0.0, near road pixels -> ~4.5. no sign flip needed; output is
only min-max normalized per image after bilinear resize to input h,w.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DAV2:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf", device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_inv(self, rgb):
        """rgb: uint8 (H,W,3) or list of them.

        returns float32 (H,W) inverse depth normalized to [0,1]
        (higher = closer), or a list matching the input order.
        """
        single = isinstance(rgb, np.ndarray)
        imgs = [rgb] if single else list(rgb)
        out = [None] * len(imgs)
        # batch only same-size groups; processor stacks into one tensor
        groups = {}
        for i, im in enumerate(imgs):
            groups.setdefault(im.shape[:2], []).append(i)
        for _, idxs in groups.items():
            batch = [imgs[i] for i in idxs]
            pix = self.processor(images=batch, return_tensors="pt")["pixel_values"]
            pix = pix.to(self.device)
            pred = self.model(pixel_values=pix).predicted_depth.float()
            for j, i in enumerate(idxs):
                h, w = imgs[i].shape[:2]
                p = F.interpolate(
                    pred[j : j + 1, None], size=(h, w), mode="bilinear", align_corners=False
                )[0, 0]
                p = p - p.min()
                p = p / p.max().clamp_min(1e-8)
                out[i] = p.cpu().numpy().astype(np.float32)
        return out[0] if single else out
