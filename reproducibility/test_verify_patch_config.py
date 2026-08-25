import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import verify_patch_config as verifier


MODEL_SOURCE = """
class MiniCPM3Model:
    def __init__(self):
        patch_size = (3, 4, 4)
        window_size = (2, 7, 7)
        self.vision_encoder_fch = SwinTransformer3D(
            patch_size=patch_size,
            window_size=window_size,
        )
        self.vision_encoder_SAX = SwinTransformer4D(
            patch_size=patch_size,
            window_size=window_size,
        )
        self.vision_encoder_LGE = SwinTransformer3D(
            patch_size=(1, 4, 4),
            window_size=(1, 7, 7),
        )
"""


def write_fake_safetensors(path: Path, tensors: dict):
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    padding = (8 - len(header) % 8) % 8
    header += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"0")


class PatchConfigVerifierTests(unittest.TestCase):
    def make_fixture(self, root: Path, kernel_rank: int = 5, omit_tensor: bool = False):
        source = root / "model.py"
        source.write_text(MODEL_SOURCE, encoding="utf-8")
        manifest = json.loads(verifier.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        shard = checkpoint / "model-00001-of-00001.safetensors"
        tensors = {}
        for encoder in manifest["encoders"]:
            in_channels = 3 if encoder["id"] == "fch" else 5
            kernel_shape = [96, in_channels, *encoder["patch_size"]]
            if kernel_rank == 4:
                kernel_shape = [96, *encoder["patch_size"]]
            tensors[encoder["checkpoint_kernel_key"]] = {
                "dtype": "BF16",
                "shape": kernel_shape,
                "data_offsets": [0, 1],
            }
            entries = verifier._window_bias_entries(encoder["window_size"])
            tensors[encoder["checkpoint_window_key"]] = {
                "dtype": "F32",
                "shape": [entries, 3],
                "data_offsets": [0, 1],
            }
        if omit_tensor:
            tensors.pop(manifest["encoders"][0]["checkpoint_window_key"])
        write_fake_safetensors(shard, tensors)
        index = {"weight_map": {key: shard.name for key in tensors}}
        (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        return source, manifest_path, checkpoint

    def test_strict_verification_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source, manifest, checkpoint = self.make_fixture(Path(temp_dir))
            report = verifier.verify(manifest, source, checkpoint, require_checkpoint=True)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(len(report["architecture_fingerprint"]), 64)

    def test_source_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source, manifest, checkpoint = self.make_fixture(Path(temp_dir))
            source.write_text(MODEL_SOURCE.replace("patch_size = (3, 4, 4)", "patch_size = (2, 4, 4)"), encoding="utf-8")
            report = verifier.verify(manifest, source, checkpoint, require_checkpoint=True)
        self.assertEqual(report["status"], "fail")
        self.assertGreaterEqual(report["summary"]["required_failures"], 2)

    def test_source_only_is_partial_not_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source, manifest, _ = self.make_fixture(Path(temp_dir))
            report = verifier.verify(manifest, source, None, require_checkpoint=False)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["summary"]["required_failures"], 0)

    def test_checkpoint_kernel_must_have_conv3d_rank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source, manifest, checkpoint = self.make_fixture(Path(temp_dir), kernel_rank=4)
            report = verifier.verify(manifest, source, checkpoint, require_checkpoint=True)
        self.assertEqual(report["status"], "fail")
        self.assertGreaterEqual(report["summary"]["required_failures"], 3)

    def test_requested_incomplete_optional_checkpoint_is_partial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source, manifest, checkpoint = self.make_fixture(Path(temp_dir), omit_tensor=True)
            report = verifier.verify(manifest, source, checkpoint, require_checkpoint=False)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["summary"]["required_failures"], 0)
        self.assertEqual(report["summary"]["optional_failures"], 1)
        self.assertEqual(report["encoders"][0]["status"], "partial")

    def test_fingerprint_is_independent_of_checkpoint_location(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_root = root / "first"
            first_root.mkdir()
            source, manifest, checkpoint_1 = self.make_fixture(first_root)
            checkpoint_2 = root / "relocated-checkpoint"
            shutil.copytree(checkpoint_1, checkpoint_2)
            first = verifier.verify(manifest, source, checkpoint_1, require_checkpoint=True)
            second = verifier.verify(manifest, source, checkpoint_2, require_checkpoint=True)
        self.assertEqual(first["architecture_fingerprint"], second["architecture_fingerprint"])


if __name__ == "__main__":
    unittest.main()
