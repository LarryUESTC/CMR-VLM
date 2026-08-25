PYTHON ?= python3
VERIFY := verify_patch_config.py
FULL_SMOKE := smoke_load_checkpoint.py
CHECKPOINT ?= $(or $(CMR_VLM_CHECKPOINT),/data/output/CLIP_biomed_all_v3_nom_seg_vstlge_fix2_Class3_nonpy_KMSCSCD_9_32)

.PHONY: verify full-smoke source-only test clean

verify:
	$(PYTHON) $(VERIFY) --checkpoint "$(CHECKPOINT)" --require-checkpoint

full-smoke:
	$(PYTHON) $(FULL_SMOKE) --checkpoint "$(CHECKPOINT)" --run-encoder-forward

source-only:
	$(PYTHON) $(VERIFY) --without-checkpoint

test:
	$(PYTHON) -m unittest -v test_verify_patch_config.py

clean:
	$(RM) outputs/patch_config.json outputs/verification_report.md outputs/full_load_smoke.json
