#!/usr/bin/env python3
"""Load the complete paper checkpoint and verify its instantiated encoders.

Unlike ``verify_patch_config.py``, this optional strong smoke test loads all
checkpoint tensor payloads through the same model class used by the archived
training and paper-evaluation entries. It never opens patient data and performs
no clinical inference.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "expected_architecture.json"
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "CMR_VLM_CHECKPOINT",
        "/data/output/CLIP_biomed_all_v3_nom_seg_vstlge_fix2_Class3_nonpy_KMSCSCD_9_32",
    )
)
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs/full_load_smoke.json"
MODEL_MODULE = "scripts.model.modeling_minicpm_solo_vst_lge6_4"
MODEL_CLASS = "MiniCPM3ForCausalLM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-encoder-forward",
        action="store_true",
        help="also execute all three encoders on small synthetic tensors (no patient data)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint.resolve()

    sys.path.insert(0, str(WORKSPACE_ROOT))
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "The full-load smoke test requires the project runtime, including torch and transformers. "
            "Use `make verify` for the standard-library metadata check."
        ) from exc

    module = importlib.import_module(MODEL_MODULE)
    model_class = getattr(module, MODEL_CLASS)

    started = time.monotonic()
    model, loading_info = model_class.from_pretrained(
        str(checkpoint),
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    model.eval()

    loading_counts = {
        name: len(loading_info.get(name, []))
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    encoders = []
    failures = []
    for expected in manifest["encoders"]:
        encoder = getattr(model.model, expected["source_attribute"])
        weight = encoder.patch_embed.proj.weight
        kernel_shape = list(weight.shape)
        actual_patch = kernel_shape[-3:]
        actual_window = list(encoder.window_size)
        checks = {
            "encoder_class": type(encoder).__name__ == expected["encoder_class"],
            "patch_size": actual_patch == expected["patch_size"],
            "window_size": actual_window == expected["window_size"],
        }
        if not all(checks.values()):
            failures.append(expected["id"])
        encoders.append(
            {
                "id": expected["id"],
                "sequence": expected["sequence"],
                "encoder_class": type(encoder).__name__,
                "kernel_shape": kernel_shape,
                "patch_size": actual_patch,
                "window_size": actual_window,
                "dtype": str(weight.dtype),
                "device": str(weight.device),
                "checks": checks,
            }
        )

    synthetic_forward = None
    if args.run_encoder_forward:
        synthetic_inputs = {
            "fch": torch.zeros((1, 3, 6, 32, 32), dtype=torch.bfloat16),
            "sax": torch.zeros((6, 5, 2, 32, 32), dtype=torch.bfloat16),
            "lge": torch.zeros((1, 5, 2, 32, 32), dtype=torch.bfloat16),
        }
        synthetic_forward = {}
        encoder_objects = {
            "fch": model.model.vision_encoder_fch,
            "sax": model.model.vision_encoder_SAX,
            "lge": model.model.vision_encoder_LGE,
        }
        with torch.inference_mode():
            for encoder_id, synthetic_input in synthetic_inputs.items():
                outputs = encoder_objects[encoder_id](synthetic_input)
                if not isinstance(outputs, (tuple, list)):
                    outputs = (outputs,)
                output_shapes = [list(output.shape) for output in outputs]
                finite = all(bool(torch.isfinite(output).all()) for output in outputs)
                if not finite:
                    failures.append(f"{encoder_id}-synthetic-forward")
                synthetic_forward[encoder_id] = {
                    "input_shape": list(synthetic_input.shape),
                    "output_shapes": output_shapes,
                    "finite": finite,
                }

    loading_clean = all(value == 0 for value in loading_counts.values())
    status = "pass" if loading_clean and not failures else "fail"
    report = {
        "status": status,
        "checkpoint": str(checkpoint),
        "model_module": MODEL_MODULE,
        "model_class": MODEL_CLASS,
        "generation_api_available": hasattr(model, "generate"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "loading_info_counts": loading_counts,
        "patient_data_accessed": False,
        "clinical_inference_performed": False,
        "synthetic_encoder_forward": synthetic_forward,
        "encoders": encoders,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{status.upper()}: loaded {report['parameter_count']:,} parameters in {report['elapsed_seconds']:.2f}s")
    print(f"Loading info: {loading_counts}")
    print(f"Generation API available: {report['generation_api_available']}")
    for encoder in encoders:
        print(
            f"{encoder['sequence']}: kernel={encoder['kernel_shape']} "
            f"patch={encoder['patch_size']} window={encoder['window_size']}"
        )
    if synthetic_forward is not None:
        for encoder_id, forward in synthetic_forward.items():
            print(
                f"{encoder_id} synthetic forward: input={forward['input_shape']} "
                f"outputs={forward['output_shapes']} finite={forward['finite']}"
            )
    print(f"JSON: {args.output_json.resolve()}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
