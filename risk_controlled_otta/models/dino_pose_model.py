from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _load_local_checkpoint(module: nn.Module, pretrained_path: str) -> None:
    if pretrained_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint = load_file(pretrained_path, device="cpu")
    else:
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    filtered = {
        key: value
        for key, value in checkpoint.items()
        if not key.startswith("head.") and not key.startswith("fc.")
    }
    incompatible = module.load_state_dict(filtered, strict=False)
    print(f"Loaded local pretrained backbone from: {pretrained_path}")
    print(f"Missing keys: {incompatible.missing_keys}")
    print(f"Unexpected keys: {incompatible.unexpected_keys}")


class DinoHeatmapHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = 11,
        mid_channels: int = 256,
        num_deconv_layers: int = 2,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        up_layers = []
        for _ in range(num_deconv_layers):
            up_layers.extend(
                [
                    nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.up = nn.Sequential(*up_layers)
        self.head = nn.Conv2d(mid_channels, num_keypoints, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x = self.proj(feature_map)
        x = self.up(x)
        return self.head(x)


class DinoHeatmapPoseModel(nn.Module):
    """
    DINO backbone + lightweight heatmap decoder.

    Default geometry:
    - input: 384 x 384
    - patch size: 16
    - patch grid: 24 x 24
    - heatmap: 48 x 48
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        input_size: int = 384,
        num_keypoints: int = 11,
        mid_channels: int = 256,
        num_deconv_layers: int = 2,
        pretrained: bool = True,
        pretrained_path: str | None = None,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for the DINO heatmap model.") from exc

        self.input_size = int(input_size)
        self.encoder = timm.create_model(
            model_name,
            pretrained=(pretrained and pretrained_path is None),
            num_classes=0,
            img_size=self.input_size,
        )
        self.num_features = getattr(self.encoder, "num_features", None)
        if self.num_features is None:
            raise ValueError(f"Unable to infer num_features from model {model_name}.")

        patch_size = getattr(self.encoder.patch_embed, "patch_size", 16)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        self.grid_size = self.input_size // self.patch_size
        self.num_grid_tokens = self.grid_size * self.grid_size
        self.num_prefix_tokens = int(
            getattr(
                self.encoder,
                "num_prefix_tokens",
                1 + int(getattr(self.encoder, "num_reg_tokens", 0)),
            )
        )
        self.decoder = DinoHeatmapHead(
            self.num_features,
            num_keypoints=num_keypoints,
            mid_channels=mid_channels,
            num_deconv_layers=num_deconv_layers,
        )

        if pretrained_path is not None:
            _load_local_checkpoint(self.encoder, pretrained_path)

    def forward_features_2d(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder.forward_features(images)

        if isinstance(tokens, dict):
            if "x_norm_patchtokens" in tokens:
                patch_tokens = tokens["x_norm_patchtokens"]
                cls_token = tokens.get("x_norm_clstoken", tokens.get("x_norm_cls_token"))
                if cls_token is None:
                    cls_token = patch_tokens.mean(dim=1)
            elif "x_prenorm" in tokens:
                x_prenorm = tokens["x_prenorm"]
                cls_token, patch_tokens = self._split_prefix_and_patch_tokens(x_prenorm)
            else:
                raise ValueError(f"Unsupported DINO forward_features dict keys: {list(tokens.keys())}")
        else:
            if isinstance(tokens, (list, tuple)):
                tokens = tokens[-1]
            if tokens.ndim != 3:
                raise ValueError(f"Unexpected token shape: {tokens.shape}")
            cls_token, patch_tokens = self._split_prefix_and_patch_tokens(tokens)

        batch_size, num_patches, channels = patch_tokens.shape
        if num_patches != self.num_grid_tokens:
            raise ValueError(f"Expected {self.num_grid_tokens} patch tokens, got {num_patches}.")

        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, channels, self.grid_size, self.grid_size)
        return feature_map, cls_token

    def _split_prefix_and_patch_tokens(self, token_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = token_tensor.shape[1]

        if num_tokens == self.num_grid_tokens:
            patch_tokens = token_tensor
            cls_token = patch_tokens.mean(dim=1)
            return cls_token, patch_tokens

        if num_tokens > self.num_grid_tokens:
            num_prefix = num_tokens - self.num_grid_tokens
            cls_token = token_tensor[:, 0] if num_prefix >= 1 else token_tensor.mean(dim=1)
            patch_tokens = token_tensor[:, num_prefix:]
            return cls_token, patch_tokens

        raise ValueError(f"Token tensor too short for {self.num_grid_tokens} patch tokens: {num_tokens}.")

    def forward(self, images: torch.Tensor, return_features: bool = False):
        feature_map, cls_token = self.forward_features_2d(images)
        heatmap = self.decoder(feature_map)
        if return_features:
            return heatmap, cls_token
        return heatmap

