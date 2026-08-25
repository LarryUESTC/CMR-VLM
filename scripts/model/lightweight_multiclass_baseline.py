from typing import Dict, Optional, Sequence

import torch
from torch import nn

from ..multimodal_encoder.video_swin_transformer import SwinTransformer3D
from ..multimodal_encoder.video_swin_transformer_4D import SwinTransformer4D


class LightweightVisionMultiClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        modalities: Sequence[str] = ("FCH", "SAX", "LGE"),
        fusion_mode: str = "concat",
        class_weights: Optional[list] = None,
    ):
        super().__init__()
        self.modalities = tuple(modality.upper() for modality in modalities)
        self.fusion_mode = fusion_mode.lower()
        if not self.modalities:
            raise ValueError("At least one modality is required.")
        if self.fusion_mode not in {"concat", "mean"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        vision_dim = 96
        depths = [1, 1, 3, 1]
        patch_size = (3, 4, 4)
        window_size = (2, 7, 7)

        feature_dims = {}

        if "FCH" in self.modalities:
            self.fch_encoder = SwinTransformer3D(
                in_chans=3,
                embed_dim=vision_dim,
                patch_size=patch_size,
                depths=depths,
                num_heads=[3, 6, 12, 24],
                window_size=window_size,
                patch_norm=True,
            )
            self.fch_encoder.init_weights()
            feature_dims["FCH"] = self.fch_encoder.num_features
            self.fch_proj = nn.Sequential(
                nn.Linear(feature_dims["FCH"], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        if "SAX" in self.modalities:
            self.sax_encoder = SwinTransformer4D(
                in_chans=5,
                embed_dim=vision_dim,
                patch_size=patch_size,
                depths=depths,
                num_heads=[3, 6, 12, 24],
                window_size=window_size,
            )
            self.sax_encoder.init_weights()
            feature_dims["SAX"] = self.sax_encoder.num_features
            self.sax_proj = nn.Sequential(
                nn.Linear(feature_dims["SAX"] * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        if "LGE" in self.modalities:
            self.lge_encoder = SwinTransformer3D(
                in_chans=5,
                embed_dim=vision_dim,
                patch_size=(1, 4, 4),
                depths=depths,
                num_heads=[3, 6, 12, 24],
                window_size=(1, 7, 7),
            )
            self.lge_encoder.init_weights()
            feature_dims["LGE"] = self.lge_encoder.num_features
            self.lge_proj = nn.Sequential(
                nn.Linear(feature_dims["LGE"], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        head_input_dim = hidden_dim * len(self.modalities) if self.fusion_mode == "concat" else hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    def set_class_weights(self, class_weights: Optional[list]) -> None:
        if class_weights is None:
            self.loss_fn = nn.CrossEntropyLoss()
            return
        device = next(self.parameters()).device
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )

    def _encode_fch(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.fch_encoder(x.float())
        return self.fch_proj(pooled)

    def _encode_sax(self, x: torch.Tensor) -> torch.Tensor:
        sax_tokens = []
        for sample in x.float():
            temporal, spatial = self.sax_encoder(sample)
            temporal = temporal.mean(dim=-1).squeeze(0)
            spatial = spatial.mean(dim=(2, 3, 4)).squeeze(0)
            sax_tokens.append(torch.cat([temporal, spatial], dim=-1))
        return self.sax_proj(torch.stack(sax_tokens, dim=0))

    def _encode_lge(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.lge_encoder(x.float())
        return self.lge_proj(pooled)

    def _collect_tokens(
        self,
        sax_vision_org_0: Optional[torch.Tensor],
        fch_vision_org: Optional[torch.Tensor],
        lge_vision_org: Optional[torch.Tensor],
    ):
        tokens = []
        for modality in self.modalities:
            if modality == "FCH":
                if fch_vision_org is None:
                    raise ValueError("FCH modality requested but input is missing.")
                tokens.append(self._encode_fch(fch_vision_org))
            elif modality == "SAX":
                if sax_vision_org_0 is None:
                    raise ValueError("SAX modality requested but input is missing.")
                tokens.append(self._encode_sax(sax_vision_org_0))
            elif modality == "LGE":
                if lge_vision_org is None:
                    raise ValueError("LGE modality requested but input is missing.")
                tokens.append(self._encode_lge(lge_vision_org))
        return tokens

    def forward(
        self,
        sax_vision_org_0: Optional[torch.Tensor] = None,
        fch_vision_org: Optional[torch.Tensor] = None,
        lge_vision_org: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = self._collect_tokens(sax_vision_org_0, fch_vision_org, lge_vision_org)
        if self.fusion_mode == "concat":
            fused = torch.cat(tokens, dim=-1)
        else:
            fused = torch.stack(tokens, dim=1).mean(dim=1)
        logits = self.head(fused)
        outputs = {"logits": logits}
        if labels is not None:
            outputs["loss"] = self.loss_fn(logits, labels)
        return outputs
