# CMR-VLM patch configuration verifier

这个核验包把论文所报告的视觉编码器 patch size 做成可独立重复的检查。它不运行临床推理，也不读取患者数据；只检查模型构造源码和 checkpoint 元数据/权重。

## 环境

- Python 3.10+
- 轻量核验（`make test` / `make verify`）只有标准库依赖，不需要 PyTorch、CUDA 或 GPU
- 完整权重装载 smoke（`make full-smoke`）需要 `torch`、`transformers` 等项目运行时（见 `requirements-smoke.txt` / `Dockerfile`）

## 正式核验

```bash
cd reproducibility
make test
make verify
```

`make verify` 默认检查论文使用的 checkpoint：

```text
CLIP_biomed_all_v3_nom_seg_vstlge_fix2_Class3_nonpy_KMSCSCD_9_32
```

可用环境变量 `CMR_VLM_CHECKPOINT` 覆盖（Docker 内默认为
`/data/output/CLIP_biomed_all_v3_nom_seg_vstlge_fix2_Class3_nonpy_KMSCSCD_9_32`）。

也可以显式指定：

```bash
python3 verify_patch_config.py \
  --checkpoint /path/to/cmr-vlm-checkpoint \
  --require-checkpoint
```

## 完整权重加载 smoke test

轻量核验只读取 safetensors header。若要进一步证明当前源码能够真正装载全部权重，可运行：

```bash
make full-smoke
```

该命令使用 `from_pretrained` 读取约 8.36 GB 的模型参数（bf16），检查
missing、unexpected、mismatched keys，并在小型合成张量上执行三路视觉 encoder。它不读取
患者数据、不执行临床推理。结果写入 `outputs/full_load_smoke.json`。

## 预期结果

- `make test`：6/6 单元测试通过；
- `make verify`：21/21 checks 通过，encoder 三路为
  `(3,4,4)/(3,4,4)/(1,4,4)`、window `(2,7,7)/(2,7,7)/(1,7,7)`；
- `make full-smoke`：`PASS`，4,177,150,750 参数，0 missing/unexpected/mismatched。

## 边界说明

这些检查只覆盖架构/权重兼容性与最小运行路径；完整病例预处理、训练、端到端报告生成和
临床指标复现需要未公开的临床数据，不属于本核验包范围。
