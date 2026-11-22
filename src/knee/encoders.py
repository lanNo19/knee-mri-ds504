"""The three frozen image encoders and their slice features (plan, Phase 4 "Models").

| name          | timm checkpoint                               | role                         |
| dinov2_vits14 | vit_small_patch14_dinov2.lvd142m (img 224)    | main comparison (ViT, SSL)   |
| convnext_tiny | convnext_tiny.fb_in22k_ft_in1k                | main comparison (CNN)        |
| vit_s16       | vit_small_patch16_224.augreg_in21k_ft_in1k    | control (ViT, supervised)    |

Slice features (768-d for every encoder, so one head fits all):
- ViTs: CLS token and the mean of the patch tokens, after the final norm.
- ConvNeXt: the pre-logits feature (global average pool, then the head LayerNorm).
Input: a grayscale uint8 slice copied to 3 channels, then the model's own
normalisation (ImageNet for DINOv2 and ConvNeXt, 0.5/0.5 for ViT-S/16).
"""
from __future__ import annotations

from dataclasses import dataclass, field

FEATURE_DIM = 768
MAIN = ("dinov2_vits14", "convnext_tiny")
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


@dataclass(frozen=True)
class Spec:
    timm_id: str
    kind: str  # "vit" or "convnext"
    norm: tuple  # (mean, std) the plan expects
    kwargs: dict = field(default_factory=dict)


SPECS = {
    "dinov2_vits14": Spec("vit_small_patch14_dinov2.lvd142m", "vit", IMAGENET, {"img_size": 224}),
    "convnext_tiny": Spec("convnext_tiny.fb_in22k_ft_in1k", "convnext", IMAGENET),
    "vit_s16": Spec("vit_small_patch16_224.augreg_in21k_ft_in1k", "vit", HALF),
}


class Encoder:
    """A frozen encoder on one device: uint8 slices in, 768-d float16 features out."""

    def __init__(self, name: str, device: str = "cpu", pretrained: bool = True, overrides: dict | None = None):
        import timm
        import torch

        spec = SPECS[name]
        self.name, self.kind, self.device = name, spec.kind, torch.device(device)
        kwargs = {**spec.kwargs, **(overrides or {})}
        model = timm.create_model(spec.timm_id, pretrained=pretrained, num_classes=0, **kwargs)
        if spec.kind == "vit":
            # the final norm must sit inside forward_features (token pooling), not after pooling
            assert isinstance(model.fc_norm, torch.nn.Identity), "unexpected ViT head layout"
        cfg = timm.data.resolve_model_data_config(model)
        mean, std = tuple(round(float(v), 3) for v in cfg["mean"]), tuple(round(float(v), 3) for v in cfg["std"])
        if (mean, std) != spec.norm:
            raise ValueError(f"{name}: normalisation differs from the plan")
        self.mean_std = (mean, std)
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self.n_prefix = int(getattr(model, "num_prefix_tokens", 1))

    def _prep(self, u8):
        """(n, 224, 224) uint8 tensor on the device -> (n, 3, 224, 224) normalised float."""
        x = u8.float().div_(255.0).unsqueeze(1)
        return (x - self._mean) / self._std

    def __call__(self, u8):
        """Slice features for a uint8 tensor (n, 224, 224) on this device; returns float16 (n, 768)."""
        import torch

        x = self._prep(u8)
        use_amp = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            if self.kind == "vit":
                t = self.model.forward_features(x)
                f = torch.cat([t[:, 0], t[:, self.n_prefix:].mean(1)], dim=1)
            else:
                f = self.model.forward_head(self.model.forward_features(x), pre_logits=True)
        return f.float().to(torch.float16)
