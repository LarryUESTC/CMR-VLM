# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch MiniCPM model."""
import math
import warnings
from typing import List, Optional, Tuple, Union, Dict
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchvision.ops.boxes import box_iou, generalized_box_iou
from torchvision.ops import generalized_box_iou_loss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_minicpm import MiniCPM3Config
import re
from .loss import BinaryDiceLoss, BCELoss
from ..multimodal_encoder.video_swin_transformer import SwinTransformer3D
from ..multimodal_encoder.video_swin_transformer_4D import SwinTransformer4D
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    pass
from .mask_decoder import MaskDecoder, MaskDecoder_sam2
from .loss_fns import sigmoid_focal_loss, dice_loss, iou_loss
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches
# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MiniCPM3Config"
IMAGE_SIZE = 512
PATCH_SIZE = 16
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# class MLP(nn.Module):
#     """简单的多层感知机"""
#
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )
#         # self.norm = MiniCPMRMSNorm(hidden_dim, eps=config.rms_norm_eps)
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x
class MLP(nn.Module):
    """简单的多层感知机"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, config):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        # 添加归一化层（假设 MiniCPMRMSNorm 已定义）
        self.norms = nn.ModuleList(
            [MiniCPMRMSNorm(hidden_dim, eps=config.rms_norm_eps) for _ in range(num_layers - 1)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = layer(x)      # 线性变换
                x = self.norms[i](x)  # 归一化
                x = F.relu(x)     # 激活函数
            else:
                x = layer(x)      # 最后一层无需激活和归一化
        return x
class DetectionHead(nn.Module):
    def __init__(self, d_model, config, num_classes=3, num_queries=3):
        """
        Args:
            d_model: Transformer输出的特征维度
            num_classes: 类别数(包括背景)
            num_queries: 查询数量(默认100，类似DETR)
        """
        super().__init__()
        self.num_queries = num_queries
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(d_model, d_model, 4, 3, config)  # 预测4个坐标(x,y,w,h)

        # 可学习的查询位置编码
        self.query_embed = nn.Embedding(num_queries, d_model)

    def forward(self, x):
        """
        Args:
            x: Transformer的输出 [batch_size, seq_len, d_model]
        Returns:
            pred_logits: 分类预测 [batch_size, num_queries, num_classes+1]
            pred_boxes: 边界框预测 [batch_size, num_queries, 4]
        """
        # 使用查询位置编码作为输入
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(x.size(0), 1, 1)

        # 简单的交叉注意力机制
        attn = torch.matmul(query_embed, x.transpose(1, 2))  # [batch, num_queries, seq_len]
        attn = F.softmax(attn, dim=-1)
        features = torch.matmul(attn, x)  # [batch, num_queries, d_model]

        # 预测分类和边界框
        # pred_logits = self.class_embed(features)
        pred_boxes_embedding = self.bbox_embed(features).sigmoid()
        # pred_boxes_embedding[..., :2] = torch.sigmoid(pred_boxes_embedding[..., :2])
        # pred_boxes_embedding[..., 2:] = torch.exp(pred_boxes_embedding[..., 2:])

        return pred_boxes_embedding


def prepare_single_target(detection_boxes, num_classes, device="cuda"):
    """
    将单张图片的 detection_boxes 转换为 DETR 的 targets 格式
    Args:
        detection_boxes: List[Dict{'class': int, 'bbox': [x_min, y_min, x_max, x_max]}]
        num_classes: 类别总数（不包括背景）
        device: 数据存放设备
    Returns:
        target: Dict{'labels': Tensor[N], 'boxes': Tensor[N, 4]}
    """
    # 提取类别和边界框
    labels = torch.tensor([box['class'] for box in detection_boxes], dtype=torch.long, device=device)
    boxes_xyxy = torch.cat([box['bbox'] for box in detection_boxes])

    # 转换为 [center_x, center_y, width, height] 并归一化
    boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy)

    return {
        'labels': labels,
        'boxes': boxes_cxcywh
    }


def xyxy_to_cxcywh(x):
    """将 [x_min, y_min, x_max, y_max] 转换为 [center_x, center_y, width, height]"""
    x_min, y_min, x_max, y_max = x.unbind(-1)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return torch.stack([center_x, center_y, width, height], dim=-1)


class DynamicDetectionLoss(nn.Module):
    def __init__(self, lambda_giou=1.0, lambda_l1=1.0, lambda_cls=0.1):
        super().__init__()
        self.lambda_giou = lambda_giou
        self.lambda_l1 = lambda_l1
        self.lambda_cls = lambda_cls

    def forward(self, pred_boxes, target_masks, target_boxes):
        """
        输入:
            pred_existence: 存在概率预测 [1, 3]
            pred_boxes: 预测边界框 [1, 3, 4]
            target_masks: 存在性标签 [1, 3] (0/1)
            target_boxes: 真实边界框 [1, 3, 4]
            image_size: 图像尺寸 (H, W)
        """
        # 分类损失（带权重的BCE）
        # cls_loss = nn.BCELoss()(pred_existence, target_masks.float()) * self.lambda_cls

        # 回归损失仅计算存在的目标
        valid_mask = target_masks.bool().squeeze()  # [3]
        target_boxes =  target_boxes.squeeze(1)  #
        # 存在目标的L1损失
        if valid_mask.any():
            l1_loss = nn.L1Loss(reduction='none')(
                pred_boxes[:, valid_mask],
                target_boxes[:, valid_mask]
            ).mean() * self.lambda_l1
        else:
            l1_loss = torch.tensor(0.0, device=pred_boxes.device)

        # GIoU损失（仅存在目标）
        if valid_mask.any():
            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[:, valid_mask])
            target_xyxy = box_cxcywh_to_xyxy(target_boxes[:, valid_mask])
            # for i in range(len(pred_xyxy)):
            #     pred_xyxy[i] = torch.clamp(pred_xyxy[i], 0, 1)
            #     target_xyxy[i] = torch.clamp(target_xyxy[i], 0, 1)
            giou_loss = generalized_box_iou_loss(pred_xyxy, target_xyxy).sum() * self.lambda_giou
        else:
            giou_loss = torch.tensor(0.0, device=pred_boxes.device)

        return l1_loss + giou_loss
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, losses, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.losses = losses
        self.eos_coef = eos_coef

        # 分类损失权重（背景类权重较低）
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, target):
        # 匹配预测和真实框
        indices = self.matcher(outputs, target)
        num_boxes = len(target['labels'])  # 真实框数量

        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f'loss_{loss}')(outputs, target, indices, num_boxes))
        return losses

    def loss_labels(self, outputs, target, indices, num_boxes):
        pred_logits = outputs[0][0]  # [num_queries, num_classes+1]
        query_idx, gt_idx = indices

        # 构建目标类别张量（未匹配的查询分配为背景类）
        target_classes = torch.full(
            pred_logits.shape[:1], self.num_classes,
            dtype=torch.long, device=pred_logits.device
        )
        target_classes[query_idx] = target['labels'][gt_idx]

        # 计算交叉熵损失
        loss_ce = F.cross_entropy(
            pred_logits, target_classes,
            weight=self.empty_weight.to(pred_logits.device)
        )
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, target, indices, num_boxes):
        pred_boxes = outputs[1][0]  # [num_queries, 4]
        query_idx, gt_idx = indices

        # 仅计算匹配的预测框和真实框的损失
        matched_pred_boxes = pred_boxes[query_idx]
        matched_gt_boxes = target['boxes'][gt_idx]

        # L1 损失
        loss_bbox = F.l1_loss(matched_pred_boxes, matched_gt_boxes, reduction='sum') / num_boxes

        # GIoU 损失
        loss_giou = (1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(matched_pred_boxes),
            box_cxcywh_to_xyxy(matched_gt_boxes)
        ))).sum() / num_boxes

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

def box_cxcywh_to_xyxy(x):
    """从(center_x, center_y, w, h)转换为(x1, y1, x2, y2)"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, target):
        """
        Args:
            outputs: Dict with keys 'pred_logits' and 'pred_boxes'
            target: Dict with keys 'labels' and 'boxes'
        Returns:
            indices: Tuple(query_indices, gt_indices)
        """
        pred_logits = outputs[0]  # [1, num_queries, num_classes+1]
        pred_boxes = outputs[1]  # [1, num_queries, 4]

        # 提取真实框和类别
        gt_labels = target['labels']  # [num_gt]
        gt_boxes = target['boxes']  # [num_gt, 4]
        num_gt = gt_boxes.shape[0]

        # 计算成本矩阵
        cost_class = -pred_logits[0, :, gt_labels]  # [num_queries, num_gt]
        cost_bbox = torch.cdist(pred_boxes[0], gt_boxes, p=1)  # [num_queries, num_gt]
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes[0]),
            box_cxcywh_to_xyxy(gt_boxes)
        )  # [num_queries, num_gt]

        # 总成本矩阵
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.cpu()  # SciPy 需要 CPU 数据

        # 匈牙利匹配
        query_indices, gt_indices = linear_sum_assignment(C)
        return (torch.as_tensor(query_indices, dtype=torch.int64),
                torch.as_tensor(gt_indices, dtype=torch.int64))


class MoEClassificationHead(nn.Module):
    def __init__(self, config, num_classes=7, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = config.hidden_size  # 2560

        # 全局特征池化层（可替换为CLS Token或其他池化方式）
        self.pooling = nn.AdaptiveAvgPool1d(1)  # [batch, tokens, 2560] → [batch, 1, 2560]

        # 专家池
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, 256),
                nn.ReLU(inplace=True),
                nn.LayerNorm(256),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_experts)
        )

        # 分类层
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # 输入尺寸: [batch, tokens, 2560]
        # x = self.pooling(x.permute(0, 2, 1)).squeeze(-1)  # [batch, 2560]

        # 门控权重计算
        gate_logits = self.gate(x)  # [batch, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)

        # 选择Top-k专家
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 并行计算专家输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch, num_experts, 256]

        # 加权融合
        batch_size = x.size(0)
        flat_topk_indices = topk_indices + torch.arange(batch_size, device=x.device).unsqueeze(1) * self.num_experts
        selected_experts = expert_outputs.view(-1, 256)[flat_topk_indices]  # [batch, top_k, 256]
        combined_output = (selected_experts * topk_weights.unsqueeze(-1)).sum(dim=1)  # [batch, 256]

        # 分类
        logits = self.classifier(combined_output)
        return logits


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.minicpm.modeling_minicpm._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.minicpm.modeling_minicpm._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.minicpm.modeling_minicpm.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )

# @torch.jit.script  # type: ignore
def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight

def show_mask(mask, ax, i, random_color=True, borders = True):
    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    color = [np.array([0 / 255, 0 / 255, 255 / 255, 0.5]),
             np.array([0 / 255, 255 / 255, 0 / 255, 0.5]),
             np.array([255 / 255, 0 / 255, 0 / 255, 0.5]),
             ][i]
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_masks(image, masks,cls_id, borders=False, text = 'No'):
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    # for i in range(0,3):
    #     mask = masks[i]
    #     show_mask(mask, plt.gca(), i, borders=borders)
    show_mask(masks, plt.gca(), cls_id, borders=borders)
    plt.title(text, fontsize=8)
    plt.axis('off')
    plt.show()
    plt.close()

def show_detection(image, outputs, target):
    # 准备绘图
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(image)
    class_names = ['LV', 'MYO','RV']
    # 绘制预测框
    pred_boxes = outputs[1].detach()[0].cpu()
    pred_classes = outputs[0].detach()[0].argmax(dim=1).cpu()
    img_h, img_w = image.size
    for box, cls_id in zip(pred_boxes, pred_classes):
        if cls_id >= len(class_names):  # 跳过背景类
            continue
        # 将归一化坐标转换为绝对坐标
        cx, cy, w, h = box.tolist()
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        # 添加矩形框和文本
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5, class_names[cls_id],
            color='lime', fontsize=12, bbox=dict(facecolor='black', alpha=0.5)
        )

    # 绘制真实框（可选）
    if 'boxes' in target:
        gt_boxes = target['boxes'].cpu()
        gt_classes = target['labels'].cpu()
        for box, cls_id in zip(gt_boxes, gt_classes):
            cx, cy, w, h = box.tolist()
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 5, class_names[cls_id],
                color='red', fontsize=12, bbox=dict(facecolor='black', alpha=0.5)
            )

    # 保存和显示
    plt.axis('off')
    plt.show()
    plt.close('all')

def denormalize_box(box, image_size):
    """将归一化的xywh坐标转换为像素级xyxy坐标"""
    # h= 300
    # w= 300
    h, w = image_size
    x_center, y_center, box_w, box_h = box
    x1 = (x_center - box_w/2) * w
    y1 = (y_center - box_h/2) * h
    x2 = (x_center + box_w/2) * w
    y2 = (y_center + box_h/2) * h
    return [x1, y1, x2, y2]

def visualize_detection(image,
                        true_boxes,
                        pred_boxes=None,
                        true_masks=None,
                        class_names=["LV", "MYO", "RV"]):
    """
    可视化检测结果

    参数:
        image_tensor: 图像张量 [C, H, W]
        true_boxes: 真实框 [3,4] (xywh)
        pred_boxes: 预测框 [3,4] (xywh)
        true_masks: 真实存在掩码 [3]
        pred_existence: 预测存在概率 [3]
        class_names: 类别名称列表
        image_size: 原始图像尺寸
        threshold: 存在性判断阈值
    """

    # 创建画布
    fig, ax = plt.subplots(1, figsize=(3, 3))
    ax.imshow(image)

    # 颜色编码 (BGR顺序便于OpenCV用户理解)
    color_map = {
        "LV": (1, 0, 0),  # 红色
        "MYO": (0, 1, 0),  # 绿色
        "RV": (0, 0, 1)  # 蓝色
    }
    # 绘制真实框
    for i, cls in enumerate(class_names):
        if true_masks is None or true_masks[i] == 1:
            box = true_boxes[i].cpu().numpy()
            if np.any(box != 0):  # 过滤占位符
                x1, y1, x2, y2 = denormalize_box(box, image.size)
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2,
                    edgecolor=color_map[cls],
                    facecolor='none',
                    linestyle='--',
                    label=f'True {cls}'
                )
                ax.add_patch(rect)

    # 绘制预测框
    if pred_boxes is not None:
        for i, cls in enumerate(class_names):
            # 存在性判断

            box = pred_boxes[i].float().cpu().numpy()
            x1, y1, x2, y2 = denormalize_box(box, image.size)

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3,
                edgecolor=color_map[cls],
                facecolor='none',
                linestyle='-',
                label=f'Pred {cls}'
            )
            ax.add_patch(rect)

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # 去重
    ax.legend(unique_labels.values(), unique_labels.keys(),
              loc='upper right', fontsize=8,
              framealpha=0.5)

    plt.axis('off')
    plt.show()
import os
def visualize_detection_test_MYO(image,
                    pred_boxes=None,
                    class_names=["LV", "MYO", "RV"],
                    save_dir="./image_output"):
    # 创建画布
    fig, ax = plt.subplots(1, figsize=(3, 3))
    ax.imshow(image)

    # 颜色编码 (BGR顺序便于OpenCV用户理解)
    color_map = {
        "LV": (1, 0, 0),  # 红色
        "MYO": (0, 1, 0),  # 绿色
        "RV": (0, 0, 1)  # 蓝色
    }

    # 绘制预测框，仅显示 MYO
    if pred_boxes is not None:
        for i, cls in enumerate(class_names):
            if cls == "LV":  # 仅处理 MYO 类别
                box = pred_boxes[i].float().cpu().numpy()
                x1, y1, x2, y2 = denormalize_box(box, image.size)

                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2,
                    edgecolor=color_map[cls],
                    facecolor='none',
                    linestyle='--',  # 使用虚线
                    label=f'Pred {cls}'
                )
                ax.add_patch(rect)

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # 去重
    ax.legend(unique_labels.values(), unique_labels.keys(),
              loc='upper right', fontsize=8,
              framealpha=0.5)

    plt.axis('off')

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存原图
    original_image_path = os.path.join(save_dir, "original_image.png")
    image.save(original_image_path)

    # 保存带框的图像
    boxed_image_path = os.path.join(save_dir, "boxed_image.png")
    plt.savefig(boxed_image_path, bbox_inches='tight', pad_inches=0)

    plt.show()
def visualize_detection_test(image,
                        pred_boxes=None,
                        class_names=["LV", "MYO", "RV"]):
    # 创建画布
    fig, ax = plt.subplots(1, figsize=(3, 3))
    ax.imshow(image)

    # 颜色编码 (BGR顺序便于OpenCV用户理解)
    color_map = {
        "LV": (1, 0, 0),  # 红色
        "MYO": (0, 1, 0),  # 绿色
        "RV": (0, 0, 1)  # 蓝色
    }
    # 绘制预测框
    if pred_boxes is not None:
        for i, cls in enumerate(class_names):
            box = pred_boxes[i].float().cpu().numpy()
            x1, y1, x2, y2 = denormalize_box(box, image.size)

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color_map[cls],
                facecolor='none',
                linestyle='-',
                label=f'Pred {cls}'
            )
            ax.add_patch(rect)

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # 去重
    ax.legend(unique_labels.values(), unique_labels.keys(),
              loc='upper right', fontsize=8,
              framealpha=0.5)

    plt.axis('off')
    plt.show()

class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniCPMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(MiniCPMRMSNorm)


class MiniCPMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            # seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MiniCPMLinearScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMDynamicNTKScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMLongRoPE(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, short_factor=None, long_factor=None, original_max_position_embeddings=None):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = (max_position_embeddings /
                 self.original_max_position_embeddings)
        self.scaling_factor = math.sqrt(
                1 + math.log(scale) /
                math.log(self.original_max_position_embeddings))
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)
        
        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype)
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype) * self.scaling_factor, persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype) * self.scaling_factor, persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)

class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class MiniCPMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniCPM3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.hidden_size // config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        self.q_a_proj = nn.Linear(
            self.hidden_size, config.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = MiniCPMRMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = MiniCPMRMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "linear":
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor = self.config.rope_scaling["factor"],
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor = self.config.rope_scaling["factor"],
                    base=self.rope_theta,
                )
            elif scaling_type == "longrope":
                self.rotary_emb = MiniCPMLongRoPE(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    short_factor = self.config.rope_scaling["short_factor"],
                    long_factor = self.config.rope_scaling["long_factor"],
                    base=self.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"]
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            if hasattr(past_key_value, "get_usable_length"):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MiniCPMFlashAttention2(MiniCPMAttention):
    """
    MiniCPM flash attention module. This module inherits from `MiniCPMAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # MiniCPMFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, "get_usable_length"):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.q_a_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in MiniCPMFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MiniCPMSdpaAttention(MiniCPMAttention):
    """
    MiniCPM attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MiniCPMAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MiniCPMAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MiniCPM3Model is using MiniCPMSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            if hasattr(past_key_value, "get_usable_length"):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
    "flash_attention_2": MiniCPMFlashAttention2,
    "sdpa": MiniCPMSdpaAttention,
}


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPM3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MINICPM_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


MINICPM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MiniCPM3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPM3PreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = MiniCPM3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniCPMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MINICPM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPM3Model(MiniCPM3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniCPMDecoderLayer`]

    Args:
        config: MiniCPM3Config
    """

    def __init__(self, config: MiniCPM3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # === Vision Patches ===
        self.M3D = False
        self.seg_enable = False
        self.use_m3d_loss = False
        self.epoch_item = 0
        self.epoch_item_text = 'epoch_i: {}'
        self.seg_text_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(0.1),
        )
        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()
        vision_patch_size = 8
        vision_patch_depth = 10
        vision_patch_depth_sd = 1
        self.hidden2D =  vision_patch_size * vision_patch_size * 3
        self.hidden3D = vision_patch_depth * vision_patch_size * vision_patch_size * 3
        self.cls_token = nn.Parameter(torch.randn(1, self.config.hidden_size))
        # hidden_size = 4096
        self.embed_vision_patch_3D = nn.Sequential(nn.Linear(
            vision_patch_depth_sd * 16 * 16 * 3,  # 32 * 32 * 3,
            self.config.hidden_size,
            bias=False,
        )) # For SD

        # === Vision Patches ===
        in_channels = 5
        vision_dim = 96
        patch_size = (3,4,4)
        depths = [1, 1, 3, 1]
        window_size = (2,7,7)
        self.vision_encoder_fch = SwinTransformer3D(
            in_chans=3,
            embed_dim=vision_dim,
            # img_size=config.img_size,
            patch_size=patch_size,
            # hidden_size=config.hidden_size,
            # mlp_dim=config.mlp_dim,
            # drop_rate = 0.1,
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=window_size,
            patch_norm=True
        )
        self.vision_encoder_fch.init_weights()

        self.vision_encoder_SAX = SwinTransformer4D(
            in_chans=in_channels,
            embed_dim=vision_dim,
            # img_size=config.img_size,
            patch_size=patch_size,
            # hidden_size=config.hidden_size,
            # mlp_dim=config.mlp_dim,
            # drop_rate = 0.1,
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=window_size,
            # patch_norm=True
        )
        self.vision_encoder_SAX.init_weights()

        self.vision_encoder_LGE = SwinTransformer3D(
            # img_size = 224,
            patch_size=(1,4,4),
            in_chans=in_channels,
            embed_dim=vision_dim,
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=(1,7,7),
        )
        self.vision_encoder_LGE.init_weights()

        self.embed_sax_vision_patch_3D_T = nn.Sequential(
            nn.Linear(self.vision_encoder_SAX.num_features, self.config.hidden_size, bias=False),
            MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )


        self.embed_sax_vision_patch_3D = nn.Sequential(
            nn.Linear(self.vision_encoder_SAX.num_features, self.config.hidden_size, bias=False),
            MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )

        self.embed_fch_vision_patch_3D = nn.Sequential(
            nn.Linear(self.vision_encoder_fch.num_features, self.config.hidden_size, bias=False),
            MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )

        self.embed_lge_vision_patch = nn.Sequential(
            nn.Linear(self.vision_encoder_LGE.num_features, self.config.hidden_size, bias=False),
            MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )


        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def text_embedding(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb
        return inputs_embeds

    def build_sdpa_mask_half(self, input_ids, inputs_embeds):
        # 找到关键位置
        b_inst_pos = (input_ids == 1).nonzero()[0, 0]
        e_inst_pos = (input_ids == 59400).nonzero()[0, 0]
        seq_len = len(input_ids)

        # 1. 初始化全"被掩码"的矩阵（使用极小数）
        mask = torch.full(
            (1, 1, seq_len, seq_len),
            torch.finfo(inputs_embeds.dtype).min,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device
        )

        # 2. 设置可关注区域
        # 所有位置可以看到图像
        mask[:, :, :, :b_inst_pos] = 1.0

        # 问题和指令部分互相可见
        q_start, q_end = b_inst_pos, e_inst_pos + 1
        mask[:, :, q_start:q_end, :q_end] = 1.0

        # 答案部分因果掩码
        a_start = e_inst_pos + 1
        for i in range(a_start, seq_len):
            mask[:, :, i, :i + 1] = 1.0

        return mask

    # 使用示例


    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        vision_patch_indices: torch.LongTensor = None,  # (batch_size, seq_length), "-1" for text token
        vision_patches: torch.FloatTensor = None,
        sax_vision_patches: torch.FloatTensor = None,  # (n_patches, 32 * 32 * 3)
        sax_vision_org_0: [torch.FloatTensor] = None,  # [(n_patches, 32 * 32 * 3)]
        sax_vision_org_1: [torch.FloatTensor] = None,  # [(n_patches, 32 * 32 * 3)]
        sax_vision_org_2: [torch.FloatTensor] = None,  # [(n_patches, 32 * 32 * 3)]
        fch_vision_patches: torch.FloatTensor = None,  # (n_patches, 32 * 32 * 3)
        lge_vision_patches: torch.FloatTensor = None,  # (n_patches, 32 * 32 * 3)
        fch_vision_org: [torch.FloatTensor] = None,  # [(n_patches, 32 * 32 * 3)]
        lge_vision_org: [torch.FloatTensor] = None,  # [(n_patches, 32 * 32 * 3)]
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if hasattr(past_key_values, "get_usable_length"):
            past_key_values_length = past_key_values.get_usable_length(seq_length)
        else:
            past_key_values_length = past_key_values.get_seq_length()

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb
            if input_ids.shape[1] != 1:
                # === Handle vision patches ===
                vision_embeds = []
                if vision_patches != None:
                    assert (vision_patch_indices.shape == input_ids.shape), "vision_patch_indices and input_ids should have the same shape"
                    vision_patches = vision_patches.squeeze()
                    vision_embeds.append(self.embed_vision_patch_3D(vision_patches))

                if fch_vision_org != None:
                    # assert (vision_patch_indices.shape == input_ids.shape), "vision_patch_indices and input_ids should have the same shape"
                    # fch_vision_patches = fch_vision_patches.squeeze()
                    fch_vision_embeds = self.vision_encoder_fch(fch_vision_org)
                    fch_vision_embeds_flat = torch.transpose(
                        torch.flatten(fch_vision_embeds[1], start_dim=2).squeeze(), 0, 1)
                    vision_embeds.append(self.embed_fch_vision_patch_3D(fch_vision_embeds_flat))

                # if sax_vision_org_0 != None:
                #     assert (vision_patch_indices.shape == input_ids.shape), "vision_patch_indices and input_ids should have the same shape"
                    # sax_vision_patches = sax_vision_patches.squeeze()

                if sax_vision_org_0 != None:
                    sax_vision_embeds_t, sax_vision_embeds_zhw = self.vision_encoder_SAX(sax_vision_org_0.squeeze())
                    sax_vision_embeds_zhw_flat = torch.transpose(torch.flatten(sax_vision_embeds_zhw, start_dim=2).squeeze(), 0, 1)
                    sax_vision_embeds_t_flat = torch.transpose(sax_vision_embeds_t.squeeze(), 0, 1)
                    vision_embeds.append(self.embed_sax_vision_patch_3D_T(sax_vision_embeds_t_flat))
                    vision_embeds.append(self.embed_sax_vision_patch_3D(sax_vision_embeds_zhw_flat))


                if lge_vision_org != None:
                    lge_vision_embeds = self.vision_encoder_LGE(lge_vision_org)
                    lge_vision_embeds_flat = torch.transpose(
                        torch.flatten(lge_vision_embeds[1], start_dim=2).squeeze(), 0, 1)
                    vision_embeds.append(self.embed_lge_vision_patch(lge_vision_embeds_flat))

                if len(vision_embeds) > 0:
                    # vision_embeds.append(self.cls_token)

                    vision_embeds.append(torch.zeros(1, self.config.hidden_size).to(vision_embeds[0].device))  # add a dummy token (for text)
                    vision_embeds = torch.cat(vision_embeds)  # (n_patches + 1, hidden_size)
                    # print(vision_embeds.size())
                    # arrange embeddings according to vision_patch_indices
                    # - text tokens are -1 (map to the dummy zero tensor)
                    # - vision tokens are 0~n_patches (map to the corresponding vision_embeds)
                    if vision_patch_indices.max() >= vision_embeds.size(0):
                        vision_patch_indices = vision_patch_indices.clamp(min=-1, max=vision_embeds.size(0) - 1)
                    vision_embeds = vision_embeds[
                        vision_patch_indices
                    ]  # (batch_size, seq_length, hidden_size)

                    # merge vision_embeds with inputs_embeds
                    inputs_embeds += vision_embeds

        # print(inputs_embeds.shape)
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            # attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            #     attention_mask,
            #     (batch_size, seq_length),
            #     inputs_embeds,
            #     past_key_values_length,
            # )
            # attention_mask = None

            try:
                attention_mask = self.build_sdpa_mask_half(input_ids[0], inputs_embeds)
            except:
                attention_mask = None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            # if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            #     print("输入包含 NaN/Inf 值！")
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MiniCPM3ForCausalLM(MiniCPM3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPM3Model(config)
        self.config = config
        # self.seg_modal = MaskDecoder(transformer_dim=config.hidden_size)#None
        # self.seg_modal = MaskDecoder_sam2(transformer_dim=config.hidden_size,
        #                                   Image_size=IMAGE_SIZE,
        #                                   Patch_size=PATCH_SIZE,
        #                                   use_high_res_features=True
        #                                   )  # None
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.add_vqaloss = True
        self.score_lv = nn.Linear(config.hidden_size, 1, bias=False)
        # self.score_cls = nn.Linear(config.hidden_size, 7, bias=False)
        config.num_labels = 7
        # self.score_cls = nn.Sequential(
        #     nn.Linear(config.hidden_size, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, config.num_labels),
        # )
        self.score_cls = MLP(config.hidden_size, config.hidden_size, config.num_labels, 2, config)

        # self.score_cls_moe = MoEClassificationHead(config, num_classes = config.num_labels)
        self.score_cls_binary = MLP(config.hidden_size, config.hidden_size, 2, 2, config)
        # Initialize weights and apply final processing
        self.detect_head = DetectionHead(d_model = config.hidden_size,config = config, num_classes=3, num_queries=3)
        self.DynamicDetectionLoss = DynamicDetectionLoss()
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        vision_patch_indices: torch.LongTensor = None,  # (batch_size, seq_length), "-1" for text token
        vision_patches: torch.FloatTensor = None,
        sax_vision_patches: torch.FloatTensor = None,
        sax_vision_org_0: torch.FloatTensor = None,
        sax_vision_org_1: torch.FloatTensor = None,
        sax_vision_org_2: torch.FloatTensor = None,
        lge_vision_patches: torch.FloatTensor = None,
        lge_vision_org: [torch.FloatTensor] = None,  # [(n_patches, 32 * 32 * 3)]
        fch_vision_patches: torch.FloatTensor = None,
        fch_vision_org: [torch.FloatTensor] = None,  # [(n_patches, 32 * 32 * 3)]
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seg_button: Optional[bool] = False,
        seg: Optional[torch.Tensor] = None,
        detection_boxes_class: Optional[List] = None,
        detection_boxes_bbox: Optional[List] = None,
        image: Optional[torch.Tensor] = None,
        text_tokens:Optional[torch.Tensor] = None,
        cls_id: Optional[int] = None,
        score_lv_button: Optional[bool] = False,
        score_lv_label: Optional[torch.LongTensor] = None,
        class_label: Optional[torch.LongTensor] = None,
            balance_loss: Optional[float] = None,
        multilabel: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniCPMForCausalLM

        >>> model = MiniCPMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_patch_indices=vision_patch_indices,
            vision_patches=vision_patches,
            sax_vision_patches=sax_vision_patches,
            sax_vision_org_0=sax_vision_org_0,
            sax_vision_org_1=sax_vision_org_1,
            sax_vision_org_2=sax_vision_org_2,
            fch_vision_patches=fch_vision_patches,
            fch_vision_org=fch_vision_org,
            lge_vision_patches=lge_vision_patches,
            lge_vision_org=lge_vision_org,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base))
        logits = logits.float()

        loss = 0
        loss_text = None
        loss_cls = None
        loss_seg = None
        loss_det = None
        loss_score = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_text = loss_fct(shift_logits, shift_labels)
            loss=loss_text

        if class_label == 'None':
            print(class_label)
            class_label = None
        if class_label is not None:
            # import pdb;pdb.set_trace()
            # self.config.pad_token_id = 4
            batch_size = input_ids.shape[0]
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:

                    sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                        hidden_states.device)
                else:
                    sequence_lengths = -1

            # input_cls = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
            input_cls = hidden_states[0].mean(dim = 0).squeeze()

            # pooled_logits = self.score_cls(input_cls)

            if len(class_label[0]) == 2:
                pooled_logits = self.score_cls_binary(input_cls)
            else:
                pooled_logits = self.score_cls(input_cls)
            # if input_ids is not None:
            #     batch_size = input_ids.shape[0]
            # else:
            #     batch_size = inputs_embeds.shape[0]
            # sequence_lengths = -1
            # print(sequence_lengths)
            # print(hidden_states.size())
            # pooled_logits = logits_cls[torch.arange(batch_size, device=logits_cls.device), sequence_lengths]

            if len(class_label[0]) == 2:
                loss_ce = BCELoss()
            elif not multilabel:
                # print('use cross!')
                loss_ce = CrossEntropyLoss()
            else:
                loss_ce = BCEWithLogitsLoss()  # 多标签分类损失函数
            # import pdb;pdb.set_trace()
            # print(loss_cls(pooled_logits.squeeze(), class_label.squeeze().float()))
            # print((torch.sigmoid(pooled_logits)> 0.5).long(), class_label)
            # pooled_logits_mean_softmax = F.softmax(pooled_logits, dim=-1)
            loss_cls = loss_ce(pooled_logits.squeeze(), class_label.squeeze().float())
            if len(class_label[0]) == 2:
                if class_label.squeeze().float()[0] == 1:
                    loss = loss_cls[0] * balance_loss + loss_cls[1] * (1 - balance_loss)
                else:
                    loss = loss_cls[1] * balance_loss + loss_cls[0] * (1 - balance_loss)
            else:
                loss = loss_cls
            # print('after:',loss)

        if score_lv_button:
            logits_lv = self.score_lv(hidden_states)
            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                        logits_lv.device
                    )
                else:
                    sequence_lengths = -1
            pooled_logits = logits_lv[torch.arange(batch_size, device=logits_lv.device), sequence_lengths]
            loss_fct = torch.nn.L1Loss()#MSELoss()
            loss_score = loss_fct(pooled_logits.squeeze(), score_lv_label.squeeze())
            loss = loss_score
        if seg_button:
            hidden_states_all = outputs[2]
            text_embedding = self.model.text_embedding(text_tokens)
            text_embedding = self.model.seg_text_projector(text_embedding.mean(1))
            low_res_masks, iou_pred, mask_tokens_out, object_score_logits = self.seg_modal(
                hidden_states.squeeze()[torch.nonzero(vision_patch_indices.squeeze() > -1).squeeze()].unsqueeze(0),
                text_embedding = text_embedding,
                high_res_features = [hidden_states_all[20].squeeze()[torch.nonzero(vision_patch_indices.squeeze() > -1).squeeze()].unsqueeze(0),
                                     hidden_states_all[40].squeeze()[torch.nonzero(vision_patch_indices.squeeze() > -1).squeeze()].unsqueeze(0)]
            )
            # maskslogits = F.interpolate(low_res_masks, size=img_shape, mode='trilinear', align_corners=False)
            maskslogits = F.interpolate(low_res_masks, (self.seg_modal.image_size,self.seg_modal.image_size), mode="bilinear", align_corners=False)

            self.model.epoch_item +=1
            if self.model.epoch_item%2000 == 1:
                image_array = image.squeeze().detach().cpu().numpy()
                image_array = np.transpose(image_array, (1, 2, 0))
                image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
                image_array = image_array.astype(np.uint8)
                image = Image.fromarray(image_array, mode='RGB')
                masks_log = maskslogits > 0.0
                masks_np = masks_log.squeeze(0).float().detach().cpu().numpy()
                text = self.model.epoch_item_text.format(self.model.epoch_item)
                show_masks(image, masks_np, cls_id = cls_id, text = text)

            if self.model.use_m3d_loss:
                loss_dice = self.model.dice_loss(maskslogits, seg)
                loss_bce = self.model.bce_loss(maskslogits, seg)
                if self.add_vqaloss:
                    loss += (loss_dice + loss_bce)
                else:
                    loss = loss_dice + loss_bce
            else:
                num_objects = 1
                loss_bce = sigmoid_focal_loss(maskslogits, seg.float(), num_objects,loss_on_multimask=True)
                loss_dice = dice_loss(maskslogits, seg.float(), num_objects, loss_on_multimask=True)
                loss_iou = iou_loss(maskslogits,seg.float(), iou_pred, num_objects,loss_on_multimask=True)

                loss_bce = loss_bce.sum()
                loss_dice = loss_dice.sum()
                loss_iou = loss_iou.sum()

                loss_seg = loss_dice + loss_bce+ loss_iou

                if self.add_vqaloss:
                    loss += loss_seg
                else:
                    loss = loss_seg

        Det_token_id = 73452
        if Det_token_id in input_ids:
            num_classes = 3
            # detection_boxes = [{'class': i, 'bbox': j} for i,j in zip(detection_boxes_class, detection_boxes_bbox)]
            # target = prepare_single_target(detection_boxes, num_classes, device="cuda")
            # matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
            # criterion = SetCriterion(num_classes, matcher, losses=['labels', 'boxes'])
            outputs_det = self.detect_head(hidden_states)
            loss = self.DynamicDetectionLoss(outputs_det, detection_boxes_class, detection_boxes_bbox)#.sum()
            # loss_dict = criterion(outputs_det, target)
            # loss = sum(loss_dict.values())
            self.model.epoch_item += 1
            if self.model.epoch_item%200 == 1:
                image_array = image.squeeze().detach().cpu().numpy()
                image_array = np.transpose(image_array, (1, 2, 0))
                image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
                image_array = image_array.astype(np.uint8)
                image = Image.fromarray(image_array, mode='RGB')
                text = self.model.epoch_item_text.format(self.model.epoch_item)
                visualize_detection(image,
                                    detection_boxes_bbox.squeeze(),
                                    pred_boxes=outputs_det.squeeze().detach(),
                                    true_masks=detection_boxes_class.squeeze(),
                                    class_names=["LV", "MYO", "RV"])
                # show_detection(image, outputs_det, detection_boxes_bbox[0])


        loss =  loss
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = getattr(past_key_values, "seen_tokens", cache_length)
                max_cache_length = past_key_values.get_max_length() if hasattr(past_key_values, "get_max_length") else None
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        # import pdb;pdb.set_trace()
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "vision_patch_indices":kwargs.get("vision_patch_indices"),
                "sax_vision_patches": kwargs.get("sax_vision_patches",None),
                "vision_patches": kwargs.get("vision_patches",None),
                "lge_vision_patches": kwargs.get("lge_vision_patches",None),
                "sax_vision_org_0": kwargs.get("sax_vision_org_0",None),
                "sax_vision_org_1": kwargs.get("sax_vision_org_1", None),
                "sax_vision_org_2": kwargs.get("sax_vision_org_2", None),
                "fch_vision_org": kwargs.get("fch_vision_org", None),
                "lge_vision_org": kwargs.get("lge_vision_org", None),
                # "fch_vision_patches": kwargs.get("fch_vision_patches", None),
                "class_label": kwargs.get("class_label",None)
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    
    @torch.inference_mode()
    def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 4096, num_beams=1, do_sample=True, top_p=0.8, temperature=0.3, logits_processor=None,
             **kwargs):
        if history is None:
            history = []
        if logits_processor:
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                        "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        else:
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                        "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        
        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(history_str, return_tensors='pt').to(self.device)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        history.append({"role": "assistant", "content": response})
        return response, history

    @torch.inference_mode()
    def forward_seg(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        vision_patch_indices: torch.LongTensor = None,  # (batch_size, seq_length), "-1" for text token
        vision_patches: torch.FloatTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image: Optional[torch.Tensor] = None,
        text_tokens:Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_patch_indices=vision_patch_indices,
            vision_patches=vision_patches,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states_all = outputs[2]

        text_embedding = self.model.text_embedding(text_tokens)
        text_embedding = self.model.seg_text_projector(text_embedding.mean(1))
        low_res_masks, iou_pred, mask_tokens_out, object_score_logits = self.seg_modal(
            hidden_states.squeeze()[torch.nonzero(vision_patch_indices.squeeze() > -1).squeeze()].unsqueeze(0),
            text_embedding = text_embedding,
            high_res_features=[
                hidden_states_all[20].squeeze()[torch.nonzero(vision_patch_indices.squeeze() > -1).squeeze()].unsqueeze(
                    0),
                hidden_states_all[40].squeeze()[torch.nonzero(vision_patch_indices.squeeze() > -1).squeeze()].unsqueeze(
                    0)]
        )
        # maskslogits = F.interpolate(low_res_masks, size=img_shape, mode='trilinear', align_corners=False)
        maskslogits = F.interpolate(low_res_masks, (IMAGE_SIZE,IMAGE_SIZE), mode="bilinear", align_corners=False)

        masks_log = maskslogits > 0.0
        masks_np = masks_log.squeeze(0).float().detach().cpu().numpy()

        return masks_np

    @torch.inference_mode()
    def forward_det(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            vision_patch_indices: torch.LongTensor = None,  # (batch_size, seq_length), "-1" for text token
            vision_patches: torch.FloatTensor = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            image: Optional[torch.Tensor] = None,
            text_tokens: Optional[torch.Tensor] = None,
            detection_boxes_class: Optional[List] = None,
            detection_boxes_bbox: Optional[List] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_patch_indices=vision_patch_indices,
            vision_patches=vision_patches,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        outputs_det = self.detect_head(hidden_states)
        self.model.epoch_item += 1
        # if self.model.epoch_item % 5 == 1:
        #     image_array = image.squeeze().detach().cpu().numpy()
        #     image_array = np.transpose(image_array, (1, 2, 0))
        #     image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
        #     image_array = image_array.astype(np.uint8)
        #     image = Image.fromarray(image_array, mode='RGB')
        #     text = self.model.epoch_item_text.format(self.model.epoch_item)
        #     visualize_detection_test_MYO(image,
        #                         # detection_boxes_bbox.squeeze(),
        #                         pred_boxes=outputs_det.squeeze().detach(),
        #                         # true_masks=detection_boxes_class.squeeze(),
        #                         class_names=["LV", "MYO", "RV"])

        # masks_log = maskslogits > 0.0
        # masks_np = masks_log.squeeze(0).float().detach().cpu().numpy()

        return outputs_det.squeeze().detach().float().cpu().numpy()

    @torch.inference_mode()
    def forward_cls(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        vision_patch_indices: torch.LongTensor = None,  # (batch_size, seq_length), "-1" for text token
        vision_patches: torch.FloatTensor = None,
        sax_vision_patches: torch.FloatTensor = None,
        lge_vision_patches: torch.FloatTensor = None,
        fch_vision_patches: torch.FloatTensor = None,
        sax_vision_org_0: torch.FloatTensor = None,
        sax_vision_org_1: torch.FloatTensor = None,
        sax_vision_org_2: torch.FloatTensor = None,
        fch_vision_org: torch.FloatTensor = None,
        lge_vision_org: [torch.FloatTensor] = None,  # [(n_patches, 32 * 32 * 3)]
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seg_button: Optional[bool] = False,
        seg: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text_tokens:Optional[torch.Tensor] = None,
        cls_id: Optional[int] = None,
        score_lv_button: Optional[bool] = False,
        score_lv_label: Optional[torch.LongTensor] = None,
        class_label: Optional[torch.LongTensor] = None,
        balance_loss: Optional[float] = None,
        multilabel: Optional[bool] = True,
    ) -> torch.LongTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_patch_indices=vision_patch_indices,
            vision_patches=vision_patches,
            sax_vision_org_0=sax_vision_org_0,
            sax_vision_org_1=sax_vision_org_1,
            sax_vision_org_2=sax_vision_org_2,
            fch_vision_patches=fch_vision_patches,
            fch_vision_org=fch_vision_org,
            lge_vision_patches=lge_vision_patches,
            lge_vision_org=lge_vision_org,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        hidden_states = outputs[0]
        # self.config.pad_token_id = 4
        batch_size = input_ids.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(hidden_states.device)
            else:
                sequence_lengths = -1

        # input_cls = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        input_cls = hidden_states.mean(dim = 1)
        # input_cls = hidden_states[0].mean(dim=0).squeeze()
        # pooled_logits = self.score_cls(input_cls)

        if len(class_label[0]) == 2:
            pooled_logits = self.score_cls_binary(input_cls)
        else:
            pooled_logits = self.score_cls(input_cls)
        # if input_ids is not None:
        #     batch_size = input_ids.shape[0]
        # else:
        #     batch_size = inputs_embeds.shape[0]
        # self.config.pad_token_id = 4
        # if self.config.pad_token_id is None and batch_size != 1:
        #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        # if self.config.pad_token_id is None:
        #     sequence_lengths = -1
        # else:
        #     if input_ids is not None:
        #         sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
        #             logits_cls.device
        #         )
        #     else:
        #         sequence_lengths = -1
        #
        # pooled_logits = logits_cls[torch.arange(batch_size, device=logits_cls.device), sequence_lengths]
        if len(class_label[0]) == 2:
            probs = F.softmax(pooled_logits, dim=1)  # 形状为 [batch_size, 2]
            # 选择概率较高的类别作为预测结果
            preds = torch.LongTensor([[0,0]]).to(probs.device)
            preds[:,torch.argmax(probs, dim=1)] = 1  # 形状为 [batch_size]
        elif not multilabel:
            probs = F.softmax(pooled_logits, dim=1)  # 形状为 [batch_size, 2]
            # 选择概率较高的类别作为预测结果
            preds = torch.LongTensor([[0]*7]).to(probs.device)
            preds[:,torch.argmax(probs, dim=1)] = 1  # 形状为 [batch_size]
        else:
            probs = torch.sigmoid(pooled_logits)  # 将 logits 转换为概率
            preds = (probs > 0.5).long()  # 通过阈值 0.5 转换为二进制标签

        return preds, pooled_logits
    @torch.inference_mode()
    def forward_lv(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        vision_patch_indices: torch.LongTensor = None,  # (batch_size, seq_length), "-1" for text token
        vision_patches: torch.FloatTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image: Optional[torch.Tensor] = None,
        text_tokens:Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_patch_indices=vision_patch_indices,
            vision_patches=vision_patches,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]


        return masks_np
@add_start_docstrings(
    """
    The MiniCPM Model transformer with a sequence classification head on top (linear layer).

    [`MiniCPMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MINICPM_START_DOCSTRING,
)
class MiniCPM3ForSequenceClassification(MiniCPM3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MiniCPM3Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
