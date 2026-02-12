import os
from typing import Optional
from dataclasses import dataclass, field

import torch
import transformers
from transformers import Trainer, AutoTokenizer

from scripts.model.modeling_minicpm_solo_vst_lge5 import MiniCPM3ForCausalLM
from src.data.instruction_tuning_CMR_v2 import AllDatasets_cls_Seg_vstlge_numpy_Mix as Causaldataset


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
    all_data_path: str = field(default="data/CMR_KM_Image_json.json", metadata={"help": "Path to data."})
    scs_data_path: str = field(default="data/CMR_SCS_Image_json.json", metadata={"help": "Path to data."})
    cd_data_path: str = field(default="data/CMR_CD_Image_json.json", metadata={"help": "Path to data."})

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

    det_km_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_Image_DET512",metadata={"help": "Path to caption data."})
    det_cd_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_CD_Image_DET512",metadata={"help": "Path to caption data."})
    det_scs_root: str = field(default="data/CMR_ALL_Image_new/CMR_ALL_SCS_Image_DET512",metadata={"help": "Path to caption data."})
    num_labels_D: int = field(default=7)

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
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 64
    eval_steps: float = 20000# 0.04
    label_names: str = "labels"
    prediction_loss_only: bool = False
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-5 #1e-4
    weight_decay: float = 0.005
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 0.001 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 8 #!!!!!!
    report_to: str = "tensorboard"
  


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

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # print(training_args)

    # tokenizer = BertTokenizer.from_pretrained(model_args.language_model_name_or_path)
    # / home / Larry / code / SOLO / data / models / MiniCPM3 - 4
    # B
    model_path = os.environ.get("MODEL_PATH", model_args.language_model_name_or_path)
    tokenizer_path = os.environ.get("TOKENIZER_PATH", model_path)
    model = MiniCPM3ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model.model.M3D = True
    # if model_args.pretrained_model:
    #     ckpt = torch.load(model_args.pretrained_model)
    #     model.load_state_dict(ckpt, strict=True)
    #     print("load pretrained model.")

    train_dataset = Causaldataset(data_args, tokenizer,  mode='train')
    eval_dataset = Causaldataset(data_args, tokenizer,  mode='validation')

    training_args._n_gpu = 1
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
    trainer.train()
    trainer.save_state()
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(training_args.output_dir, 'model_params.bin'))


if __name__ == "__main__":
    main()
