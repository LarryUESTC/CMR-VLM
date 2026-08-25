from typing import Dict, Optional

import torch
from torch import nn

from ..multimodal_encoder.video_swin_transformer import SwinTransformer3D
from ..multimodal_encoder.video_swin_transformer_4D import SwinTransformer4D


class VisionTransformerMultiClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        hidden_dim: int = 512,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        dropout: float = 0.1,
        class_weights: Optional[list] = None,
    ):
        super().__init__()
        vision_dim = 96
        depths = [1, 1, 3, 1]
        patch_size = (3, 4, 4)
        window_size = (2, 7, 7)

        self.fch_encoder = SwinTransformer3D(
            in_chans=3,
            embed_dim=vision_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=window_size,
            patch_norm=True,
        )
        self.sax_encoder = SwinTransformer4D(
            in_chans=5,
            embed_dim=vision_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=window_size,
        )
        self.lge_encoder = SwinTransformer3D(
            in_chans=5,
            embed_dim=vision_dim,
            patch_size=(1, 4, 4),
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=(1, 7, 7),
        )

        self.fch_encoder.init_weights()
        self.sax_encoder.init_weights()
        self.lge_encoder.init_weights()

        fused_vision_dim = self.fch_encoder.num_features
        self.fch_proj = nn.Sequential(nn.Linear(fused_vision_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.sax_temporal_proj = nn.Sequential(nn.Linear(fused_vision_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.sax_spatial_proj = nn.Sequential(nn.Linear(fused_vision_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.lge_proj = nn.Sequential(nn.Linear(fused_vision_dim, hidden_dim), nn.LayerNorm(hidden_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.modality_embed = nn.Parameter(torch.randn(1, 5, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=fusion_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
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
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    def _encode_fch(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.fch_encoder(x.float())
        return self.fch_proj(pooled)

    def _encode_sax(self, x: torch.Tensor):
        temporal_tokens = []
        spatial_tokens = []
        for sample in x.float():
            temporal, spatial = self.sax_encoder(sample)
            temporal_tokens.append(temporal.mean(dim=-1).squeeze(0))
            spatial_tokens.append(spatial.mean(dim=(2, 3, 4)).squeeze(0))
        temporal = torch.stack(temporal_tokens, dim=0)
        spatial = torch.stack(spatial_tokens, dim=0)
        return self.sax_temporal_proj(temporal), self.sax_spatial_proj(spatial)

    def _encode_lge(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.lge_encoder(x.float())
        return self.lge_proj(pooled)

    def forward(
        self,
        sax_vision_org_0: torch.Tensor,
        fch_vision_org: torch.Tensor,
        lge_vision_org: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fch_token = self._encode_fch(fch_vision_org)
        sax_temporal_token, sax_spatial_token = self._encode_sax(sax_vision_org_0)
        lge_token = self._encode_lge(lge_vision_org)

        tokens = torch.stack([fch_token, sax_temporal_token, sax_spatial_token, lge_token], dim=1)
        cls_token = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.modality_embed[:, : tokens.size(1)]

        fused = self.fusion(tokens)
        logits = self.head(fused[:, 0])
        outputs = {"logits": logits}
        if labels is not None:
            outputs["loss"] = self.loss_fn(logits, labels)
        return outputs
