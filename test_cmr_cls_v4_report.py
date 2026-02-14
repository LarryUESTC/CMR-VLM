import os
import json
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import jieba
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_score,
    recall_score, f1_score, roc_auc_score
)
from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score

from scripts.model.modeling_minicpm_solo_vst_lge5 import MiniCPM3ForCausalLM
from src.data.instruction_tuning_CMR_v2 import AllDatasets_cls_Seg_vstlge_numpy_Cap_Rep_test as VQADataset

B_INST, E_INST = "[INST]", "[/INST]"
DEVICE = "cuda:1"

def calculate_bertscore(pred, ref):
    P, R, F1 = bert_score([pred], [ref], lang="zh", model_type="/home/Larry/code/SOLO/src/bert-base-chinese",num_layers=12)
    return F1.mean().item()

@dataclass
class DataArguments:
    data_root: str = field(default="/home/Larry/data/CMR_KM_Image", metadata={"help": "Root directory for all data."})
    scs_root: str = field(default="/home/Larry/data/CMR_SCS_Image", metadata={"help": "Root directory for all data."})
    cd_root: str = field(default="/home/Larry/data/CMR_Chendu_Image", metadata={"help": "Root directory for all data."})
    all_data_path: str = field(default="/home/Larry/data/CMR_KM_Image_json.json", metadata={"help": "Path to data."})
    scs_data_path: str = field(default="/home/Larry/data/CMR_SCS_Image_json.json", metadata={"help": "Path to data."})
    cd_data_path: str = field(default="/home/Larry/data/CMR_CD_Image_json.json", metadata={"help": "Path to data."})
    sd_data_path3D: str = field(default="/home/Larry/data/CMR_NCSD_SD_0.json",
                                metadata={"help": "Path to caption data."})
    location_data_path3D: str = field(default="/home/Larry/data/CMR_NCSD_location.json",
                                      metadata={"help": "Path to caption data."})
    max_length: int = field(default=4096)
    seg_Lv_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_Image_seg/LV",
                             metadata={"help": "Path to caption data."})
    seg_Rv_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_Image_seg/RV",
                             metadata={"help": "Path to caption data."})
    seg_MYO_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_Image_seg/MYO",
                              metadata={"help": "Path to caption data."})
    seg_scs_Lv_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_SCS_Image_seg/LV",
                                 metadata={"help": "Path to caption data."})
    seg_scs_Rv_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_SCS_Image_seg/RV",
                                 metadata={"help": "Path to caption data."})
    seg_scs_MYO_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_SCS_Image_seg/MYO",
                                  metadata={"help": "Path to caption data."})
    seg_cd_Lv_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_CD_Image_seg/LV",
                                metadata={"help": "Path to caption data."})
    seg_cd_Rv_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_CD_Image_seg/RV",
                                metadata={"help": "Path to caption data."})
    seg_cd_MYO_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_CD_Image_seg/MYO",
                                 metadata={"help": "Path to caption data."})
    det_km_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_Image_DET512",
                             metadata={"help": "Path to caption data."})
    det_cd_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_CD_Image_DET512",
                             metadata={"help": "Path to caption data."})
    det_scs_root: str = field(default="/home/Larry/data/CMR_ALL_Image_new/CMR_ALL_SCS_Image_DET512",
                              metadata={"help": "Path to caption data."})
    num_labels_D: int = field(default=7)


class VQAInferencePipeline:
    def __init__(self, model_path: str, device: str = DEVICE):
        self.model_path = model_path
        self.device = device
        self._setup_directories()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = self._load_model()
        self.metrics = {
            "BLEU": [],
            "METEOR": [],
            "ROUGE1": [],
            "ROUGE2": [],
            "ROUGEL": [],
            "BERTScore": []
        }

    def _setup_directories(self):
        os.makedirs(os.path.join(self.model_path, 'runs/'), exist_ok=True)

    def _load_model(self) -> MiniCPM3ForCausalLM:
        model = MiniCPM3ForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.model.M3D = True
        try:
            model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        except Exception:
            model.generation_config = GenerationConfig.from_model_config(model.config)
        model = model.to(self.device)
        return model

    @staticmethod
    def extract_text_after_inst(text: str) -> str:
        match = re.search(r"\[INST\].*?\[\/INST\](.*)", text, re.DOTALL)
        return match.group(1).strip() if match else text

    def prepare_dataloader(self, data_args: DataArguments) -> torch.utils.data.DataLoader:
        test_dataset = VQADataset(
            data_args,
            self.tokenizer,
            mode='test'
        )

        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

    def generate_outputs(
            self,
            inputs: dict[str, torch.Tensor],
            generation_config = None
    ) -> str:
        default_config = {
            "do_sample": False,
            "top_p": 0.95,
            "max_length": 4096,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if generation_config:
            default_config.update(generation_config)

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            vision_patch_indices=inputs["vision_patch_indices"],
            sax_vision_patches=inputs.get("sax_vision_patches"),
            fch_vision_patches=inputs.get("fch_vision_patches"),
            lge_vision_patches=inputs.get("lge_vision_patches"),
            sax_vision_org_0=inputs.get("sax_vision_org_0"),
            sax_vision_org_1=inputs.get("sax_vision_org_1"),
            sax_vision_org_2=inputs.get("sax_vision_org_2"),
            fch_vision_org=inputs.get("fch_vision_org"),
            lge_vision_org=inputs.get("lge_vision_org"),
            **default_config
        )

        text_out = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self.extract_text_after_inst(text_out)

    def process_batch(self, batch):
        inputs = {
            "input_ids": batch["input_ids"].to(device=self.device),
            "attention_mask": batch["attention_mask"].to(device=self.device),
            "vision_patch_indices": batch["vision_patch_indices"].to(device=self.device),
        }

        for key in ["sax_vision_patches", "fch_vision_patches", "lge_vision_patches",
                    "sax_vision_org_0", "sax_vision_org_1", "sax_vision_org_2",
                    "fch_vision_org", "lge_vision_org"]:
            if batch.get(key) is not None:
                inputs[key] = batch[key].to(device=self.device)

        return inputs

    def calculate_metrics(self, prediction: str, reference: str):
        hyp_tokens = list(jieba.cut(prediction))
        ref_tokens = [list(jieba.cut(ref.replace('<|im_end|>', ''))) for ref in [reference]]
        bleu = sentence_bleu(ref_tokens, hyp_tokens)
        meteor = meteor_score(ref_tokens, hyp_tokens)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        rouge_scores = scorer.score(" ".join(jieba.cut(reference)), " ".join(hyp_tokens))
        bert_f1 = calculate_bertscore(prediction, reference)

        return {
            "BLEU": bleu,
            "METEOR": meteor,
            "ROUGE1": rouge_scores['rouge1'].fmeasure,
            "ROUGE2": rouge_scores['rouge2'].fmeasure,
            "ROUGEL": rouge_scores['rougeL'].fmeasure,
            "BERTScore": bert_f1
        }

    def run_inference(
            self,
            do_sample: bool = False,
            top_p: float = 0.95,
            max_new_tokens: int = 30
    ):
        parser = HfArgumentParser(DataArguments)
        data_args = parser.parse_args_into_dataclasses()[0]

        test_dataloader = self.prepare_dataloader(data_args)
        results = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                inputs = self.process_batch(batch)
                output_text = self.generate_outputs(inputs, {
                    "do_sample": do_sample,
                    "top_p": top_p,
                })

                reference_text = batch["text"][0].replace('<|im_end|>', '')
                question = batch["question"][0]

                metrics = self.calculate_metrics(output_text, reference_text)
                for key in self.metrics:
                    self.metrics[key].append(metrics[key])

                result = {
                    "question": question,
                    "reference": reference_text,
                    "prediction": output_text,
                    **metrics
                }

                print("========== Output ==========")
                print(f"Question: {question}")
                print(f"Reference: {reference_text}")
                print(f"Prediction: {output_text}")
                print("\nEvaluation Metrics:")
                for name, value in metrics.items():
                    print(f"{name}: {value:.4f}")
                print("============================")

                results.append(result)

                if len(results) % 20 == 0:
                    self.save_results(results)
                    self.print_intermediate_metrics()

        self.save_results(results)
        self.print_final_metrics()
        return results

    def print_intermediate_metrics(self):
        print("\nIntermediate Metrics Averages:")
        for name, values in self.metrics.items():
            print(f"{name}: {np.mean(values):.4f}")

    def print_final_metrics(self):
        print("\nFinal Evaluation Metrics:")
        for name, values in self.metrics.items():
            print(f"{name} (avg): {np.mean(values):.4f}")

    def save_results(self, results) -> None:
        output_path = os.path.join(self.model_path, 'runs', 'report_inference_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


def main():
    MODEL_PATH = "/home/Larry/code/SOLO/output/CLIP_biomed_all_v3_nom_seg_vstlge_fix5_CRQ"

    pipeline = VQAInferencePipeline(MODEL_PATH)
    results = pipeline.run_inference(
        do_sample=False,
        top_p=0.95,
    )


if __name__ == "__main__":
    main()
