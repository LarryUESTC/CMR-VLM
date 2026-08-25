#!/usr/bin/env python3
"""Verify the executed CMR-VLM patch configuration without loading model weights.

The verifier uses only the Python standard library. It inspects the actual model
constructor with ``ast`` and reads only the JSON headers of safetensors shards.
No patient data, PyTorch installation, GPU, or tensor payload loading is needed.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "expected_architecture.json"
DEFAULT_MODEL_SOURCE = WORKSPACE_ROOT / "scripts/model/modeling_minicpm_solo_vst_lge6_4.py"
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "CMR_VLM_CHECKPOINT",
        "/data/output/CLIP_biomed_all_v3_nom_seg_vstlge_fix2_Class3_nonpy_KMSCSCD_9_32",
    )
)
DEFAULT_JSON_OUTPUT = SCRIPT_DIR / "outputs/patch_config.json"
DEFAULT_MARKDOWN_OUTPUT = SCRIPT_DIR / "outputs/verification_report.md"


class VerificationError(RuntimeError):
    """Raised when a requested verification input cannot be interpreted."""


def _json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Unable to read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"Expected a JSON object in {path}")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_ast_value(node: ast.AST, environment: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name):
        if node.id not in environment:
            raise VerificationError(f"Unresolved constructor variable: {node.id}")
        return environment[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_resolve_ast_value(item, environment) for item in node.elts]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_resolve_ast_value(node.operand, environment)
    raise VerificationError(f"Unsupported constructor expression: {ast.dump(node, include_attributes=False)}")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def inspect_model_constructor(model_source: Path, model_class: str) -> dict[str, dict[str, Any]]:
    """Return explicitly instantiated encoder parameters from the model constructor."""
    try:
        source_text = model_source.read_text(encoding="utf-8")
    except OSError as exc:
        raise VerificationError(f"Unable to read model source {model_source}: {exc}") from exc

    try:
        tree = ast.parse(source_text, filename=str(model_source))
    except SyntaxError as exc:
        raise VerificationError(f"Unable to parse model source {model_source}: {exc}") from exc

    class_node = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == model_class),
        None,
    )
    if class_node is None:
        raise VerificationError(f"Class {model_class} was not found in {model_source}")
    init_node = next(
        (node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"),
        None,
    )
    if init_node is None:
        raise VerificationError(f"Class {model_class} has no __init__ method")

    environment: dict[str, Any] = {}
    for node in init_node.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                environment[target.id] = _resolve_ast_value(node.value, environment)
            except VerificationError:
                continue

    encoders: dict[str, dict[str, Any]] = {}
    for node in ast.walk(init_node):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.value, ast.Call):
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr.startswith("vision_encoder_")
        ):
            continue
        keyword_map = {keyword.arg: keyword.value for keyword in node.value.keywords if keyword.arg}
        if "patch_size" not in keyword_map or "window_size" not in keyword_map:
            continue
        encoders[target.attr] = {
            "encoder_class": _call_name(node.value.func),
            "patch_size": list(_resolve_ast_value(keyword_map["patch_size"], environment)),
            "window_size": list(_resolve_ast_value(keyword_map["window_size"], environment)),
            "constructor_line": node.lineno,
            "patch_line": keyword_map["patch_size"].lineno,
            "window_line": keyword_map["window_size"].lineno,
        }
    return encoders


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read a safetensors JSON header without reading tensor payloads."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise VerificationError(f"Invalid safetensors file (missing header length): {path}")
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length <= 1 or header_length > file_size - 8 or header_length > 256 * 1024 * 1024:
                raise VerificationError(f"Implausible safetensors header length {header_length} in {path}")
            header_bytes = handle.read(header_length)
    except OSError as exc:
        raise VerificationError(f"Unable to read safetensors header from {path}: {exc}") from exc
    try:
        return json.loads(header_bytes.decode("utf-8").rstrip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Invalid safetensors JSON header in {path}: {exc}") from exc


class CheckpointMetadata:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        self._headers: dict[Path, dict[str, Any]] = {}
        self.index_path: Path | None = None
        self.weight_map: dict[str, str] = {}

        if checkpoint.is_file():
            if checkpoint.suffix != ".safetensors":
                raise VerificationError(f"Checkpoint file must be a .safetensors file: {checkpoint}")
            self.single_file = checkpoint
            return

        self.single_file = None
        if not checkpoint.is_dir():
            raise VerificationError(f"Checkpoint path does not exist: {checkpoint}")
        self.index_path = checkpoint / "model.safetensors.index.json"
        if not self.index_path.exists():
            self.index_path = None
        if self.index_path is not None:
            index = _json_load(self.index_path)
            self.weight_map = index.get("weight_map", {})
            if not isinstance(self.weight_map, dict):
                raise VerificationError(f"Invalid weight_map in {self.index_path}")
        else:
            candidate = checkpoint / "model.safetensors"
            if not candidate.exists():
                raise VerificationError(
                    f"No model.safetensors or model.safetensors.index.json was found in {checkpoint}; "
                    "this metadata-only verifier supports safetensors checkpoints only"
                )
            self.single_file = candidate

    def _header(self, path: Path) -> dict[str, Any]:
        if path not in self._headers:
            self._headers[path] = read_safetensors_header(path)
        return self._headers[path]

    def tensor(self, key: str) -> dict[str, Any]:
        if self.single_file is not None:
            shard = self.single_file
        else:
            shard_name = self.weight_map.get(key)
            if shard_name is None:
                raise VerificationError(f"Tensor {key} is absent from {self.index_path}")
            shard = self.checkpoint / shard_name
        if shard.suffix != ".safetensors":
            raise VerificationError(f"Tensor {key} points to a non-safetensors shard: {shard}")
        metadata = self._header(shard).get(key)
        if not isinstance(metadata, dict) or "shape" not in metadata:
            raise VerificationError(f"Tensor {key} is absent from safetensors header {shard}")
        return {
            "key": key,
            "shard": str(shard.resolve()),
            "dtype": metadata.get("dtype"),
            "shape": metadata.get("shape"),
        }


def _window_bias_entries(window_size: list[int]) -> int:
    value = 1
    for width in window_size:
        value *= 2 * int(width) - 1
    return value


def _check(name: str, expected: Any, actual: Any, required: bool = True, detail: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "required": required,
        "status": "pass" if expected == actual else "fail",
        "detail": detail,
    }


def verify(
    manifest_path: Path,
    model_source: Path,
    checkpoint: Path | None,
    require_checkpoint: bool = False,
) -> dict[str, Any]:
    manifest = _json_load(manifest_path)
    source_values = inspect_model_constructor(model_source, manifest["model_class"])
    checkpoint_reader: CheckpointMetadata | None = None
    checkpoint_error: str | None = None
    if checkpoint is not None:
        try:
            checkpoint_reader = CheckpointMetadata(checkpoint)
        except VerificationError as exc:
            checkpoint_error = str(exc)

    encoder_results: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []
    fingerprint_material: dict[str, Any] = {
        "manifest": manifest,
        "model_source_sha256": _sha256_file(model_source),
        "checkpoint_tensors": {},
    }

    for expected in manifest["encoders"]:
        source = source_values.get(expected["source_attribute"])
        checks: list[dict[str, Any]] = []
        if source is None:
            checks.append(_check("source constructor", "present", "missing"))
            source = {}
        else:
            checks.extend(
                [
                    _check("encoder class", expected["encoder_class"], source.get("encoder_class")),
                    _check("source patch size", expected["patch_size"], source.get("patch_size")),
                    _check("source window size", expected["window_size"], source.get("window_size")),
                ]
            )

        checkpoint_result: dict[str, Any] = {
            "requested": checkpoint is not None,
            "available": checkpoint_reader is not None,
            "path": str(checkpoint.resolve()) if checkpoint is not None else None,
            "error": checkpoint_error,
        }
        if checkpoint_reader is not None:
            try:
                kernel = checkpoint_reader.tensor(expected["checkpoint_kernel_key"])
                window = checkpoint_reader.tensor(expected["checkpoint_window_key"])
                checkpoint_result.update({"kernel": kernel, "window_bias": window})
                actual_patch = list(kernel["shape"][-3:]) if len(kernel["shape"]) >= 3 else list(kernel["shape"])
                expected_entries = _window_bias_entries(expected["window_size"])
                actual_entries = window["shape"][0] if window["shape"] else None
                checks.extend(
                    [
                        _check("checkpoint patch-kernel rank", 5, len(kernel["shape"])),
                        _check("checkpoint patch kernel", expected["patch_size"], actual_patch),
                        _check("checkpoint window-bias rank", 2, len(window["shape"])),
                        _check("checkpoint window-bias entries", expected_entries, actual_entries),
                    ]
                )
                fingerprint_material["checkpoint_tensors"][expected["id"]] = {
                    "kernel": {
                        "key": kernel["key"],
                        "dtype": kernel["dtype"],
                        "shape": kernel["shape"],
                    },
                    "window_bias": {
                        "key": window["key"],
                        "dtype": window["dtype"],
                        "shape": window["shape"],
                    },
                }
            except VerificationError as exc:
                checkpoint_result["error"] = str(exc)
                checks.append(_check("checkpoint tensors", "present", "missing", required=require_checkpoint, detail=str(exc)))
        else:
            checks.append(
                {
                    "name": "checkpoint verification",
                    "expected": "available" if require_checkpoint else "optional",
                    "actual": "unavailable",
                    "required": require_checkpoint,
                    "status": "fail" if require_checkpoint else "skip",
                    "detail": checkpoint_error or "No checkpoint path was supplied.",
                }
            )

        all_checks.extend(checks)
        encoder_results.append(
            {
                **expected,
                "source": source,
                "checkpoint": checkpoint_result,
                "checks": checks,
                "status": "fail" if any(item["status"] == "fail" and item["required"] for item in checks) else (
                    "partial" if any(item["status"] in {"skip", "fail"} for item in checks) else "pass"
                ),
            }
        )

    required_failures = [item for item in all_checks if item["required"] and item["status"] == "fail"]
    skipped = [item for item in all_checks if item["status"] == "skip"]
    optional_failures = [item for item in all_checks if not item["required"] and item["status"] == "fail"]
    status = "fail" if required_failures else ("partial" if skipped or optional_failures else "pass")
    fingerprint_json = json.dumps(fingerprint_material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": manifest.get("schema_version", "1.0"),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "require_checkpoint": require_checkpoint,
        "architecture_fingerprint": _sha256_bytes(fingerprint_json.encode("utf-8")),
        "inputs": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256_file(manifest_path),
            "model_source": str(model_source.resolve()),
            "model_source_sha256": fingerprint_material["model_source_sha256"],
            "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
        },
        "summary": {
            "encoders": len(encoder_results),
            "checks": len(all_checks),
            "passed": sum(item["status"] == "pass" for item in all_checks),
            "failed": sum(item["status"] == "fail" for item in all_checks),
            "skipped": len(skipped),
            "required_failures": len(required_failures),
            "optional_failures": len(optional_failures),
        },
        "encoders": encoder_results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CMR-VLM patch configuration verification",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Generated: `{report['generated_at']}`",
        f"- Architecture fingerprint: `{report['architecture_fingerprint']}`",
        f"- Manifest: `{report['inputs']['manifest']}`",
        f"- Manifest SHA-256: `{report['inputs']['manifest_sha256']}`",
        f"- Model source: `{report['inputs']['model_source']}`",
        f"- Model source SHA-256: `{report['inputs']['model_source_sha256']}`",
        f"- Checkpoint: `{report['inputs']['checkpoint'] or 'not supplied'}`",
        "",
        "| Encoder | Patch size (axes) | Window size (axes) | Source | Checkpoint |",
        "|---|---|---|---|---|",
    ]
    for encoder in report["encoders"]:
        source_ok = all(
            check["status"] == "pass" for check in encoder["checks"] if check["name"].startswith("source") or check["name"] == "encoder class"
        )
        checkpoint_checks = [check for check in encoder["checks"] if check["name"].startswith("checkpoint")]
        checkpoint_state = "PASS" if checkpoint_checks and all(check["status"] == "pass" for check in checkpoint_checks) else (
            "SKIP" if checkpoint_checks and all(check["status"] == "skip" for check in checkpoint_checks) else "FAIL"
        )
        patch = " × ".join(map(str, encoder["patch_size"]))
        patch_axes = ", ".join(encoder["patch_axes"])
        window = " × ".join(map(str, encoder["window_size"]))
        window_axes = ", ".join(encoder["window_axes"])
        lines.append(
            f"| {encoder['sequence']} | `{patch}` ({patch_axes}) | `{window}` ({window_axes}) | "
            f"{'PASS' if source_ok else 'FAIL'} | {checkpoint_state} |"
        )
    lines.extend(["", "## Detailed checks", ""])
    for encoder in report["encoders"]:
        lines.append(f"### {encoder['sequence']}")
        lines.append("")
        lines.append(encoder["interpretation"])
        lines.append("")
        source = encoder.get("source", {})
        if source:
            lines.append(
                f"- Constructor: `{encoder['source_attribute']}` at line `{source.get('constructor_line')}` "
                f"(patch argument line `{source.get('patch_line')}`, window argument line `{source.get('window_line')}`)"
            )
        checkpoint = encoder.get("checkpoint", {})
        kernel = checkpoint.get("kernel")
        window_bias = checkpoint.get("window_bias")
        if kernel:
            lines.append(
                f"- Patch kernel: `{kernel['key']}`, shape `{kernel['shape']}`, dtype `{kernel['dtype']}`, "
                f"shard `{Path(kernel['shard']).name}`"
            )
        if window_bias:
            lines.append(
                f"- Window-bias table: `{window_bias['key']}`, shape `{window_bias['shape']}`, "
                f"dtype `{window_bias['dtype']}`, shard `{Path(window_bias['shard']).name}`"
            )
        if source or kernel or window_bias:
            lines.append("")
        for check in encoder["checks"]:
            lines.append(
                f"- **{check['status'].upper()}** — {check['name']}: expected `{check['expected']}`, observed `{check['actual']}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], json_output: Path, markdown_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--without-checkpoint", action="store_true", help="Run source-only verification.")
    parser.add_argument("--require-checkpoint", action="store_true", help="Fail if checkpoint metadata cannot be verified.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = None if args.without_checkpoint else args.checkpoint
    try:
        report = verify(
            manifest_path=args.manifest.resolve(),
            model_source=args.model_source.resolve(),
            checkpoint=checkpoint.resolve() if checkpoint is not None else None,
            require_checkpoint=args.require_checkpoint,
        )
        write_report(report, args.output_json.resolve(), args.output_markdown.resolve())
    except VerificationError as exc:
        print(f"verification error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(
            f"{report['status'].upper()}: {report['summary']['passed']}/{report['summary']['checks']} checks passed; "
            f"fingerprint={report['architecture_fingerprint'][:16]}"
        )
        print(f"JSON: {args.output_json.resolve()}")
        print(f"Markdown: {args.output_markdown.resolve()}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
