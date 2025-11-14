# CMR-VLM

🫀 CMR-VLM: Spatiotemporal Vision-Language Foundation Model for 4D Cardiovascular MR Interpretation

CMR-VLM is the first vision–language foundation model natively designed for multi-sequence 4D cardiac MRI. It unifies the full clinical CMR interpretation workflow—from sequence/phase/slice recognition and reference-guided segmentation to abnormality VQA, disease classification, and structured report generation—within a single, instruction-driven architecture.

✨ Key Features

✅ 15 clinical tasks in one model: spans perception → reasoning → diagnosis → reporting

✅ Native 4D spatiotemporal modeling: specialized 2D/3D/4D encoders for LGE, 4CH, and SAX

✅ Instruction-driven flexibility: respond to natural language queries like “Segment the LV” or “Is there LGE in the septum?”

✅ Missing-modality robust: enables contrast-free screening with cine-only inputs

✅ Clinically efficient: generates reports in 4.2 seconds (∼400× faster than experts)

✅ Single-GPU trainable: 4B-parameter model, no massive clusters needed

🏆 State-of-the-Art Performance

+136% VQA accuracy over SOTA medical VLMs

Dice > 0.93 for LV segmentation (vs. nnUNet)

AUC 0.937 (internal) / 0.860 (external) in 16-class CVD diagnosis

🧩 Built on MiniCPM-4B + Swin Transformer, trained on 4,058 multi-center CMR cases with standardized reports.
