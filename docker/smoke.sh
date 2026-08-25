#!/usr/bin/env bash
# CMR-VLM container smoke-test driver. All steps are patient-free.
#
# Steps 3/4 require the released checkpoint. When the checkpoint is not mounted
# they are SKIPPED (exit 0), unless CMR_VLM_REQUIRE_CHECKPOINT=1 is set, in
# which case a missing checkpoint is a failure.
set -euo pipefail

SKIP_FULL_LOAD="${CMR_VLM_SKIP_FULL_LOAD:-1}"
REQUIRE_CHECKPOINT="${CMR_VLM_REQUIRE_CHECKPOINT:-0}"
CHECKPOINT="${CMR_VLM_CHECKPOINT:-/data/output/CLIP_biomed_all_v3_nom_seg_vstlge_fix2_Class3_nonpy_KMSCSCD_9_32}"

echo "=== environment ==="
python -c "import sys, torch, transformers, huggingface_hub; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('transformers', transformers.__version__); print('huggingface_hub', huggingface_hub.__version__)"
echo

echo "=== 1/4 python demo.py ==="
python demo.py | tee /tmp/demo_out.txt
grep -Fq "torch.Size([1, 16, 256])" /tmp/demo_out.txt
echo "PASS: demo.py returned the expected tensor shape torch.Size([1, 16, 256])"
echo

echo "=== 2/4 verifier unit tests ==="
cd reproducibility
python -m unittest -v test_verify_patch_config.py
echo

if [ -d "$CHECKPOINT" ] || [ -f "$CHECKPOINT" ]; then
    echo "=== 3/4 verifier checkpoint-header checks ==="
    python verify_patch_config.py --checkpoint "$CHECKPOINT" --require-checkpoint
    echo
else
    if [ "$REQUIRE_CHECKPOINT" = "1" ]; then
        echo "FAIL: checkpoint not mounted at $CHECKPOINT"
        exit 1
    fi
    echo "=== 3/4 verifier checkpoint-header checks: SKIPPED (checkpoint not mounted; mount the released checkpoint at /data/output to run) ==="
    echo
fi

if [ "$SKIP_FULL_LOAD" = "1" ]; then
    echo "=== 4/4 full checkpoint load: SKIPPED (set CMR_VLM_SKIP_FULL_LOAD=0 to enable; needs ~8.4 GB RAM) ==="
else
    if [ -d "$CHECKPOINT" ] || [ -f "$CHECKPOINT" ]; then
        echo "=== 4/4 full checkpoint load + synthetic encoder forward ==="
        python smoke_load_checkpoint.py --checkpoint "$CHECKPOINT" --run-encoder-forward
    else
        if [ "$REQUIRE_CHECKPOINT" = "1" ]; then
            echo "FAIL: checkpoint not mounted at $CHECKPOINT"
            exit 1
        fi
        echo "=== 4/4 full checkpoint load: SKIPPED (checkpoint not mounted) ==="
    fi
fi
echo
echo "ALL CMR-VLM SMOKE TESTS PASSED"
