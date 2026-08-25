import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import torch
import transformers
from torch.utils.data import ConcatDataset, Subset
from transformers import Trainer, AutoTokenizer

from scripts.model.modeling_minicpm_solo_vst_lge5 import MiniCPM3ForCausalLM
from src.data.instruction_tuning_CMR_v2 import (
    AllDatasets_cls_Seg_vstlge_numpy_Mix as Causaldataset,
    AllDatasets_cls_Seg_vstlge_numpy_new2 as ClassificationDataset,
)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
LOCAL_DATA_ROOTS = [
    Path(os.environ["CMR_DATA_ROOT"]).expanduser()
    for _ in [0]
    if os.environ.get("CMR_DATA_ROOT")
]
LOCAL_DATA_ROOTS.append(Path("/home/Larry/data"))


def _iter_candidate_paths(raw_path: str):
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        yield path
        return

    for root in (Path.cwd(), WORKSPACE_ROOT, SCRIPT_DIR):
        yield (root / path).resolve()

    if path.parts and path.parts[0] == "data":
        suffix = Path(*path.parts[1:])
        for data_root in LOCAL_DATA_ROOTS:
            yield (data_root / suffix).resolve()


def resolve_existing_path(raw_path: Optional[str], label: str) -> Optional[str]:
    if raw_path is None:
        return None

    checked = []
    seen = set()
    for candidate in _iter_candidate_paths(raw_path):
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        checked.append(candidate_str)
        if candidate.exists():
            return candidate_str

    checked_paths = "\n  - ".join(checked)
    raise FileNotFoundError(
        f"Could not resolve {label}: {raw_path}\n"
        f"Tried:\n  - {checked_paths}"
    )


def log_stage(message: str) -> None:
    print(f"[train] {message}", flush=True)


CRITICAL_LOADING_PREFIXES = (
    "model.embed_vision_patch",
    "model.embed_vision_patch_3D",
    "model.embed_fch_vision_patch_3D",
    "model.embed_lge_vision_patch",
    "model.embed_sax_vision_patch_3D",
    "model.embed_sax_vision_patch_3D_T",
    "model.vision_encoder_fch",
    "model.vision_encoder_SAX",
    "model.vision_encoder_LGE",
    "score_cls",
    "score_cls_binary",
)


def _filter_loading_keys(keys):
    return [key for key in keys if key.startswith(CRITICAL_LOADING_PREFIXES)]


def _format_loading_key_sample(keys, limit=8):
    if not keys:
        return "none"
    sample = list(keys[:limit])
    if len(keys) > limit:
        sample.append("...")
    return ", ".join(sample)


def validate_loading_info(loading_info, checkpoint_path: str, allow_partial_model_init: bool) -> None:
    critical_missing = _filter_loading_keys(loading_info.get("missing_keys", []))
    critical_unexpected = _filter_loading_keys(loading_info.get("unexpected_keys", []))
    critical_mismatched = _filter_loading_keys(
        [item.get("key", "") for item in loading_info.get("mismatched_keys", [])]
    )

    if allow_partial_model_init:
        log_stage(
            "Partial multimodal checkpoint initialization is allowed; "
            f"critical missing={len(critical_missing)}, unexpected={len(critical_unexpected)}, "
            f"mismatched={len(critical_mismatched)}"
        )
        return

    if critical_missing or critical_unexpected or critical_mismatched:
        raise RuntimeError(
            "Checkpoint/model mismatch detected before training. "
            f"Checkpoint: {checkpoint_path}\n"
            f"Critical missing keys ({len(critical_missing)}): {_format_loading_key_sample(critical_missing)}\n"
            f"Critical unexpected keys ({len(critical_unexpected)}): {_format_loading_key_sample(critical_unexpected)}\n"
            f"Critical mismatched keys ({len(critical_mismatched)}): {_format_loading_key_sample(critical_mismatched)}\n"
            "This training entry uses the local multimodal model definition; "
            "do not continue unless you intentionally want random or mismatched visual weights. "
            "Pass --allow_partial_model_init True only if you have verified the checkpoint compatibility yourself."
        )


def move_sample_to_device(sample, model):
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            value = value.unsqueeze(0)
            if value.dtype.is_floating_point:
                batch[key] = value.to(device=device, dtype=model_dtype)
            else:
                batch[key] = value.to(device=device)
        else:
            batch[key] = value
    return batch


def run_multimodal_sanity_check(model, dataset, num_samples: int = 3) -> None:
    was_training = model.training
    model.eval()
    try:
        checked = 0
        for idx in range(min(len(dataset), num_samples)):
            sample = dataset[idx]
            has_vision = any(
                key in sample for key in ("sax_vision_org_0", "fch_vision_org", "lge_vision_org")
            )
            if not has_vision:
                continue
            batch = move_sample_to_device(sample, model)
            with torch.no_grad():
                outputs = model(**batch)
            if outputs.loss is not None and not torch.isfinite(outputs.loss).all():
                raise RuntimeError(
                    f"Multimodal sanity check failed at sample {idx}: non-finite loss detected."
                )
            if hasattr(outputs, "logits") and outputs.logits is not None and not torch.isfinite(outputs.logits).all():
                raise RuntimeError(
                    f"Multimodal sanity check failed at sample {idx}: non-finite logits detected."
                )
            checked += 1
        if checked == 0:
            raise RuntimeError("Multimodal sanity check did not find any sample with vision inputs.")
    finally:
        model.train(was_training)


def prepare_model_runtime_device(model) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    return device


def resolve_runtime_paths(model_args, data_args):
    model_args.language_model_name_or_path = resolve_existing_path(
        model_args.language_model_name_or_path, "language_model_name_or_path"
    )
    if model_args.pretrained_model:
        model_args.pretrained_model = resolve_existing_path(
            model_args.pretrained_model, "pretrained_model"
        )

    data_path_fields = [
        "data_root",
        "scs_root",
        "cd_root",
        "YA_root",
        "all_data_path",
        "scs_data_path",
        "cd_data_path",
        "YA_data_path",
        "sd_data_path3D",
        "location_data_path3D",
        "seg_Lv_root",
        "seg_Rv_root",
        "seg_MYO_root",
        "seg_scs_Lv_root",
        "seg_scs_Rv_root",
        "seg_scs_MYO_root",
        "seg_cd_Lv_root",
        "seg_cd_Rv_root",
        "seg_cd_MYO_root",
        "seg_YA_Lv_root",
        "seg_YA_Rv_root",
        "seg_YA_MYO_root",
        "det_km_root",
        "det_cd_root",
        "det_scs_root",
        "det_YA_root",
    ]
    for field_name in data_path_fields:
        setattr(
            data_args,
            field_name,
            resolve_existing_path(getattr(data_args, field_name), field_name),
        )

    return model_args, data_args


# 你的训练代码...
@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    language_model_name_or_path: str = field(default="data/models/SoloMiniCPM3-4B-UNI-V4")

    gather_loss: bool = field(default=False, metadata={"help": "Gather all distributed batch data of multiple GPUs and calculate contrastive loss together."})
    local_loss: bool = field(default=False)

    pretrained_model: str = field(default=None)
    in_channels: int = field(default=3)
    img_size: tuple = field(default=(224, 224))
    patch_size: tuple = field(default=(16, 16)) #

    num_labels: int = field(default=7)
    # mlp_dim: int = field(default=3072)
    # num_layers: int = field(default=12)
    # num_heads: int = field(default=12)
    # pos_embed: str = field(default="perceptron")
    # dropout_rate: float = field(default=0.0)
    # spatial_dims: int = field(default=3)
    # max_text_len: int = field(default=4096)
    # vocab_size: int = field(default=30522)


@dataclass
class DataArguments:
    data_root: str = field(default="data/CMR_KM_Image", metadata={"help": "Root directory for all data."})
    scs_root: str = field(default="data/CMR_SCS_Image", metadata={"help": "Root directory for all data."})
    cd_root: str = field(default="data/CMR_Chendu_Image", metadata={"help": "Root directory for all data."})
    YA_root: str = field(default="data/CMR_YA_Image", metadata={"help": "Root directory for YA external data."})
    all_data_path: str = field(default="data/CMR_KM_Image_json.json", metadata={"help": "Path to data."})
    scs_data_path: str = field(default="data/CMR_SCS_Image_json.json", metadata={"help": "Path to data."})
    cd_data_path: str = field(default="data/CMR_CD_Image_json.json", metadata={"help": "Path to data."})
    YA_data_path: str = field(default="data/CMR_YA_Image_json.json", metadata={"help": "Path to YA external data."})

    sd_data_path3D: str = field(default="data/CMR_NCSD_SD_0.json",metadata={"help": "Path to caption data."})
    location_data_path3D: str = field(default="data/CMR_NCSD_location.json",metadata={"help": "Path to caption data."})
    max_length: int = field(default=4096)
    seg_Lv_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_Image_seg/LV",metadata={"help": "Path to caption data."})
    seg_Rv_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_Image_seg/RV",metadata={"help": "Path to caption data."})
    seg_MYO_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_Image_seg/MYO",metadata={"help": "Path to caption data."})
    seg_scs_Lv_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_SCS_Image_seg/LV",
                             metadata={"help": "Path to caption data."})
    seg_scs_Rv_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_SCS_Image_seg/RV",
                             metadata={"help": "Path to caption data."})
    seg_scs_MYO_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_SCS_Image_seg/MYO",
                              metadata={"help": "Path to caption data."})
    seg_cd_Lv_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_CD_Image_seg/LV",
                             metadata={"help": "Path to caption data."})
    seg_cd_Rv_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_CD_Image_seg/RV",
                             metadata={"help": "Path to caption data."})
    seg_cd_MYO_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_CD_Image_seg/MYO",
                              metadata={"help": "Path to caption data."})
    seg_YA_Lv_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_YA_Image_seg/LV",
                             metadata={"help": "Path to YA segmentation data."})
    seg_YA_Rv_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_YA_Image_seg/RV",
                             metadata={"help": "Path to YA segmentation data."})
    seg_YA_MYO_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_YA_Image_seg/MYO",
                              metadata={"help": "Path to YA segmentation data."})

    det_km_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_Image_DET512",metadata={"help": "Path to caption data."})
    det_cd_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_CD_Image_DET512",metadata={"help": "Path to caption data."})
    det_scs_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_SCS_Image_DET512",metadata={"help": "Path to caption data."})
    det_YA_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_YA_Image_DET512", metadata={"help": "Path to YA detection data."})
    exclude_diagnoses: str = field(
        default="",
        metadata={"help": "Comma-separated diagnoses to exclude from dataset construction, e.g. DCM,MI"},
    )
    num_labels_D: int = field(default=7)
    task_mode: str = field(
        default="mix",
        metadata={"help": "Dataset composition. Supported: mix, classification_only"},
    )
    classification_repeat_factor: int = field(
        default=5,
        metadata={"help": "Repeat factor for classification datasets inside mix mode."},
    )
    classification_single_label_only: bool = field(
        default=False,
        metadata={"help": "If True, keep only single-label classification samples after diagnosis exclusion."},
    )
    classification_train_max_samples: int = field(
        default=0,
        metadata={"help": "If >0, randomly downsample only the training classification dataset to this many samples."},
    )
    classification_train_sample_seed: int = field(
        default=42,
        metadata={"help": "Seed for classification_train_max_samples downsampling."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora_enable: bool = False
    # lora_r: int = 64  # 16
    # lora_alpha: int = 64  # 32
    # lora_dropout: float = 0.05  # 0.1
    # lora_weight_path: str = ""
    # lora_bias: str = "none"
    tune_vision: Optional[bool] = field(default=False)
    tune_llm: Optional[bool] = field(default=True)
    # llm_type: str = field(default="minicpm")
    # use_lora: Optional[bool] = field(default=False)
    # max_slice_nums: Optional[int] = field(default=9)

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adafactor")
    remove_unused_columns: bool = field(default=False)
    allow_partial_model_init: bool = field(default=False)
    skip_multimodal_sanity_check: bool = field(default=False)


    # ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    # dataloader_drop_last: bool = True
    # debug: bool = True
    # config in bash file
    bf16: bool = True
    output_dir: str = "./output/CLIP_biomed_all_v3_nom_seg_vstlge_fix2_mix_ALL"
    # use_cpu: bool=True ####!!!!!
    num_train_epochs: int = 5 #30!!!
    _n_gpu: int = 1
    per_device_train_batch_size: int = 1 #32
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    eval_strategy: str = "steps"
    eval_accumulation_steps: int = 64
    eval_steps: int = 2000
    label_names: str = "labels"
    prediction_loss_only: bool = False
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    learning_rate: float = 1e-5 #1e-4
    weight_decay: float = 0.005
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 0.001 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 8 #!!!!!!
    report_to: str = "tensorboard"
    classification_pooling: str = field(default="mean")
    use_class_weights: bool = field(default=False)
    classification_weight_scheme: str = field(
        default="inverse",
        metadata={"help": "Class-weight transform for single-label classification: none, inverse, sqrt, log."},
    )
    classification_weight_clip_max: float = field(
        default=0.0,
        metadata={"help": "Optional upper bound for transformed class weights. <=0 disables clipping."},
    )
    classification_weight_normalize: bool = field(
        default=True,
        metadata={"help": "Normalize non-zero class weights to mean 1 after transform/clipping."},
    )
    classification_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Optional label smoothing applied to single-label classification loss."},
    )
    classification_pair_margin_classes: str = field(
        default="",
        metadata={"help": "Optional comma-separated class names for a targeted pairwise margin, e.g. '肥厚型心肌病,心肌梗死'."},
    )
    classification_pair_margin_value: float = field(
        default=0.0,
        metadata={"help": "Margin value used for targeted pairwise classification separation. <=0 disables it."},
    )
    classification_pair_margin_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the targeted pairwise margin loss. <=0 disables it."},
    )
    mix_classification_loss_reweight: bool = field(
        default=True,
        metadata={"help": "Allow classification class weights to be applied in mix mode when use_class_weights is enabled."},
    )
  


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    correct = (preds == labels).sum()
    total = labels.size
    acc = correct / total
    return {"accuracy": acc}

def preprocess_logits_for_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return preds


def build_dataset(data_args, tokenizer, mode: str):
    if data_args.task_mode == "classification_only":
        return ClassificationDataset(data_args, tokenizer, mode=mode)
    if data_args.task_mode == "mix":
        return Causaldataset(data_args, tokenizer, mode=mode)
    raise ValueError(f"Unsupported task_mode: {data_args.task_mode}")


def _resolve_dataset_class_names(dataset):
    class_names = getattr(dataset, "CLASSES_CN", None)
    if class_names:
        return class_names
    if isinstance(dataset, Subset):
        return _resolve_dataset_class_names(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        for sub_dataset in dataset.datasets:
            class_names = _resolve_dataset_class_names(sub_dataset)
            if class_names:
                return class_names
        return None
    if hasattr(dataset, "dataset"):
        return _resolve_dataset_class_names(dataset.dataset)
    return None


def _accumulate_class_counts_from_dataset(dataset, class_counts: torch.Tensor) -> bool:
    valid_idx = getattr(dataset, "valid_idx", None)
    data_list = getattr(dataset, "data_list", None)
    class_names = getattr(dataset, "CLASSES_CN", None)
    if (
        getattr(dataset, "prompt_mode", None) == "classification"
        and valid_idx is not None
        and data_list is not None
        and class_names is not None
    ):
        found_any = False
        for data_idx in valid_idx:
            try:
                class_name = data_list[data_idx]["classes"]
            except Exception:
                continue
            try:
                class_ind = class_names.index(class_name)
            except ValueError:
                continue
            if class_ind >= class_counts.numel():
                continue
            class_counts[class_ind] += 1
            found_any = True
        return found_any

    if isinstance(dataset, ConcatDataset):
        found_any = False
        for sub_dataset in dataset.datasets:
            found_any = _accumulate_class_counts_from_dataset(sub_dataset, class_counts) or found_any
        return found_any

    if hasattr(dataset, "dataset"):
        return _accumulate_class_counts_from_dataset(dataset.dataset, class_counts)

    return False


def compute_classification_class_weights(
    dataset,
    num_labels: int,
    scheme: str = "inverse",
    clip_max: float = 0.0,
    normalize: bool = True,
):
    class_counts = torch.zeros(num_labels, dtype=torch.float64)

    if not _accumulate_class_counts_from_dataset(dataset, class_counts):
        for idx in range(len(dataset)):
            sample = dataset[idx]
            class_label = sample.get("class_label")
            if not torch.is_tensor(class_label):
                continue
            class_label = class_label.detach().cpu()
            if class_label.numel() != num_labels:
                continue
            if class_label.sum().item() <= 0:
                continue
            class_idx = int(class_label.argmax().item())
            class_counts[class_idx] += 1

    total = class_counts.sum().item()
    if total <= 0:
        raise RuntimeError("Failed to compute class weights: no labeled classification samples were found.")

    scheme = (scheme or "inverse").strip().lower()
    if scheme not in {"none", "inverse", "sqrt", "log"}:
        raise ValueError(
            f"Unsupported classification_weight_scheme: {scheme}. "
            "Expected one of: none, inverse, sqrt, log."
        )

    weights = torch.zeros(num_labels, dtype=torch.float64)
    nonzero_mask = class_counts > 0
    if not nonzero_mask.any():
        raise RuntimeError("Failed to compute class weights: all class counts are zero.")

    active_classes = int(nonzero_mask.sum().item())
    raw_inverse = total / (active_classes * class_counts[nonzero_mask])

    if scheme == "none":
        transformed = torch.ones_like(raw_inverse)
    elif scheme == "sqrt":
        transformed = torch.sqrt(raw_inverse)
    elif scheme == "log":
        transformed = torch.log1p(raw_inverse)
    else:
        transformed = raw_inverse

    if clip_max > 0:
        transformed = torch.clamp(transformed, max=clip_max)

    if normalize and transformed.numel() > 0:
        transformed = transformed / transformed.mean().clamp(min=1e-12)

    weights[nonzero_mask] = transformed
    return weights.tolist(), class_counts.tolist()


def configure_classification_pair_margin(model, train_dataset, training_args) -> None:
    pair_spec = (training_args.classification_pair_margin_classes or "").strip()
    margin_value = float(training_args.classification_pair_margin_value)
    margin_weight = float(training_args.classification_pair_margin_weight)

    if not pair_spec or margin_value <= 0.0 or margin_weight <= 0.0:
        model.config.classification_pair_margin_indices = None
        model.config.classification_pair_margin_class_names = None
        model.config.classification_pair_margin_value = 0.0
        model.config.classification_pair_margin_weight = 0.0
        log_stage(
            "Classification pair margin: disabled "
            f"(classes={pair_spec or 'none'}, margin={margin_value}, weight={margin_weight})"
        )
        return

    class_names = _resolve_dataset_class_names(train_dataset)
    if not class_names:
        raise ValueError("Unable to configure classification pair margin because dataset class names are unavailable.")

    pair_names = [item.strip() for item in pair_spec.split(",") if item.strip()]
    if len(pair_names) != 2:
        raise ValueError(
            "classification_pair_margin_classes must contain exactly two comma-separated class names. "
            f"Received: {pair_spec}"
        )

    missing = [name for name in pair_names if name not in class_names]
    if missing:
        raise ValueError(
            "classification_pair_margin_classes contains unknown labels: "
            f"{missing}. Available classes: {list(class_names)}"
        )

    pair_indices = [int(class_names.index(name)) for name in pair_names]
    model.config.classification_pair_margin_indices = pair_indices
    model.config.classification_pair_margin_class_names = pair_names
    model.config.classification_pair_margin_value = margin_value
    model.config.classification_pair_margin_weight = margin_weight
    log_stage(
        "Classification pair margin: "
        f"enabled classes={pair_names} indices={pair_indices} "
        f"margin={margin_value} weight={margin_weight}"
    )

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args = resolve_runtime_paths(model_args, data_args)
    if training_args.logging_nan_inf_filter:
        log_stage("Disabling logging_nan_inf_filter so NaN/Inf losses are surfaced instead of being masked as 0.0.")
        training_args.logging_nan_inf_filter = False
    # print(training_args)

    # tokenizer = BertTokenizer.from_pretrained(model_args.language_model_name_or_path)
    # / home / Larry / code / SOLO / data / models / MiniCPM3 - 4
    # B
    model_path = resolve_existing_path(
        os.environ.get("MODEL_PATH", model_args.language_model_name_or_path), "MODEL_PATH"
    )
    tokenizer_path = resolve_existing_path(
        os.environ.get("TOKENIZER_PATH", model_path), "TOKENIZER_PATH"
    )
    log_stage(f"Model path: {model_path}")
    log_stage(f"Tokenizer path: {tokenizer_path}")
    log_stage(f"KM data root: {data_args.data_root}")
    log_stage(f"Task mode: {data_args.task_mode}")
    log_stage(f"Exclude diagnoses: {data_args.exclude_diagnoses or 'none'}")
    log_stage(f"Classification single-label only: {data_args.classification_single_label_only}")
    log_stage(f"Mix classification repeat factor: {data_args.classification_repeat_factor}")
    log_stage("Loading model and tokenizer...")
    model, loading_info = MiniCPM3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_loading_info=True,
    )
    validate_loading_info(loading_info, model_path, training_args.allow_partial_model_init)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model.model.M3D = True
    model.config.classification_pooling = training_args.classification_pooling
    model.config.classification_cls_token_id = tokenizer.convert_tokens_to_ids("<CLS>")
    model.config.classification_label_smoothing = max(
        0.0,
        float(training_args.classification_label_smoothing),
    )
    log_stage(
        f"Classification pooling: {training_args.classification_pooling} "
        f"(cls_token_id={model.config.classification_cls_token_id})"
    )
    log_stage(
        f"Classification label smoothing: {model.config.classification_label_smoothing}"
    )
    # if model_args.pretrained_model:
    #     ckpt = torch.load(model_args.pretrained_model)
    #     model.load_state_dict(ckpt, strict=True)
    #     print("load pretrained model.")

    log_stage("Building train dataset...")
    train_dataset = build_dataset(data_args, tokenizer, mode='train')
    if (
        data_args.task_mode == "classification_only"
        and data_args.classification_train_max_samples
        and data_args.classification_train_max_samples > 0
        and data_args.classification_train_max_samples < len(train_dataset)
    ):
        generator = torch.Generator()
        generator.manual_seed(int(data_args.classification_train_sample_seed))
        selected_indices = torch.randperm(len(train_dataset), generator=generator)[
            : int(data_args.classification_train_max_samples)
        ].tolist()
        train_dataset = Subset(train_dataset, selected_indices)
        log_stage(
            "Applied train-only random downsampling: "
            f"max_samples={data_args.classification_train_max_samples} "
            f"seed={data_args.classification_train_sample_seed}"
        )
    log_stage(f"Train dataset ready: {len(train_dataset)} samples")
    log_stage("Building validation dataset...")
    eval_dataset = build_dataset(data_args, tokenizer, mode='validation')
    log_stage(f"Validation dataset ready: {len(eval_dataset)} samples")

    weight_scheme = (training_args.classification_weight_scheme or "inverse").strip().lower()
    apply_class_weights = training_args.use_class_weights and weight_scheme != "none" and (
        data_args.task_mode == "classification_only"
        or (data_args.task_mode == "mix" and training_args.mix_classification_loss_reweight)
    )
    if apply_class_weights:
        log_stage("Computing classification class weights...")
        class_weights, class_counts = compute_classification_class_weights(
            train_dataset,
            model.config.num_labels,
            scheme=weight_scheme,
            clip_max=training_args.classification_weight_clip_max,
            normalize=training_args.classification_weight_normalize,
        )
        model.config.classification_class_weights = class_weights
        log_stage(
            "Classification class weighting: "
            f"enabled scheme={weight_scheme} "
            f"clip_max={training_args.classification_weight_clip_max} "
            f"normalize={training_args.classification_weight_normalize}"
        )
        log_stage(f"Classification class counts: {class_counts}")
        log_stage(f"Classification class weights: {class_weights}")
    else:
        model.config.classification_class_weights = None
        log_stage(
            "Classification class weighting: disabled "
            f"(use_class_weights={training_args.use_class_weights}, "
            f"scheme={weight_scheme}, "
            f"task_mode={data_args.task_mode})"
        )
    configure_classification_pair_margin(model, train_dataset, training_args)

    if data_args.task_mode == "classification_only":
        training_args.label_names = ["labels", "class_label"]
        if training_args.load_best_model_at_end and training_args.metric_for_best_model == "eval_loss":
            log_stage(
                "classification_only mode does not produce a stable eval_loss metric for best-model selection; "
                "disabling load_best_model_at_end so checkpoints can still be saved during evaluation."
            )
            training_args.load_best_model_at_end = False
            training_args.metric_for_best_model = None
    log_stage(
        f"Validation is configured every {training_args.eval_steps} steps; "
        f"best model selection uses {training_args.metric_for_best_model}."
    )

    log_stage("Preparing runtime device...")
    runtime_device = prepare_model_runtime_device(model)
    log_stage(f"Moved model to runtime device: {runtime_device}")

    if not training_args.skip_multimodal_sanity_check:
        log_stage("Running multimodal sanity check on train samples...")
        run_multimodal_sanity_check(model, train_dataset)
        log_stage("Multimodal sanity check passed.")

    training_args._n_gpu = 1
    log_stage("Initializing Trainer...")
    trainer = Trainer(
                        model=model,
                        args=training_args,
                        # data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,

                        # compute_metrics=compute_metrics,
                        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                      )

    # if you want to resume your training, pls set the checkpoint in trainer.train(resume_from_checkpoint="")
    # trainer.train(resume_from_checkpoint="output/CLIP_biomed_all_v3_nom_seg_vstlge_fix2_mix/checkpoint-32000")
    log_stage("Starting training...")
    trainer.train()
    trainer.save_state()
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(training_args.output_dir, 'model_params.bin'))


if __name__ == "__main__":
    main()
