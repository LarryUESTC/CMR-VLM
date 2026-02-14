# CMR-VLM

🫀 CMR-VLM: Spatiotemporal Vision-Language Foundation Model for 4D Cardiovascular MR Interpretation

CMR-VLM is a vision–language foundation model designed for multi-sequence 4D cardiac MRI. It unifies the clinical CMR interpretation workflow—from sequence/phase/slice recognition and reference-guided segmentation to abnormality VQA, disease classification, and structured report generation—within a single, instruction-driven architecture.

## Key Features

- 15 clinical tasks in one model, spanning perception, reasoning, diagnosis, and reporting
- Native 4D spatiotemporal modeling with specialized 2D/3D/4D encoders for LGE, 4CH, and SAX
- Instruction-driven flexibility for natural language prompts
- Missing-modality robustness enabling contrast-free screening with cine-only inputs
- Clinically efficient inference with fast report generation
- Single-GPU trainable 4B-parameter model

## Performance Highlights

- +136% VQA accuracy over SOTA medical VLMs
- Dice > 0.93 for LV segmentation (vs. nnUNet)
- AUC 0.937 (internal) / 0.860 (external) in 16-class CVD diagnosis

## System Requirements

- Linux
- Python 3.10
- CUDA-enabled GPU recommended for training and full inference
- CPU-only is sufficient for the demo script

## Installation

```bash
conda env create -f environment.yml
conda activate solo
pip install -r requirements.txt
```

## Data

- Data paths are configured in [train_CLIP_biomed_cmp4b_ALL_V3_loss_seg_vstlge_fix_mix.py](train_CLIP_biomed_cmp4b_ALL_V3_loss_seg_vstlge_fix_mix.py) via `DataArguments`.
- Environment variables are supported for Excel and model paths.
- A minimal text QA sample is available at [qa_sample.json](data/sample/qa_sample.json) for format reference.

## Demo

The demo performs a lightweight, text-only forward pass with a small configuration.

```bash
python demo.py
```

Ensure huggingface-hub is within [0.34.0, 1.0.0).

## Training

```bash
MODEL_PATH=path/to/model TOKENIZER_PATH=path/to/tokenizer \
python train_CLIP_biomed_cmp4b_ALL_V3_loss_seg_vstlge_fix_mix.py
```

## Evaluation

Evaluation uses the same training entry point with evaluation mode and dataset configuration defined in `DataArguments`.

## Reproducibility

- Dependencies are pinned in [requirements.txt](requirements.txt).
- Model configuration is defined in [configuration_minicpm.py](scripts/model/configuration_minicpm.py).
- Training arguments are tracked in [train_CLIP_biomed_cmp4b_ALL_V3_loss_seg_vstlge_fix_mix.py](train_CLIP_biomed_cmp4b_ALL_V3_loss_seg_vstlge_fix_mix.py).

## License

Apache-2.0 in [LICENSE](LICENSE).
