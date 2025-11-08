# AUDIT CHECKLIST - Action Items

## Critical Path (Do These First)

- [ ] **1. Validate Learning Rate** (1 hour)
  - Location: `scripts/phase4-fine-tune-model/06_train_mlx.py` line 31
  - Current: `LEARNING_RATE = 5e-6`
  - Action: Test with 1e-5, monitor validation loss
  - Reference: AUDIT_REPORT.md Section 2.1

- [ ] **2. Add Training Metrics Capture** (2 hours)
  - Location: `scripts/phase4-fine-tune-model/06_train_mlx.py` lines 413-447
  - Action: Parse subprocess output for loss curves
  - Capture: steps, train_loss, val_loss → save to JSON
  - Reference: AUDIT_REPORT.md Section 3.1, code example in AUDIT_SUMMARY.md

- [ ] **3. Create Model Evaluation Script** (1-2 hours)
  - Location: Create `scripts/phase4-fine-tune-model/06b_evaluate_finetuned.py`
  - Action: Test fine-tuned model on domain questions
  - Measure: Response length, formatting compliance, style adherence
  - Reference: AUDIT_REPORT.md Section 3.1, code template provided

- [ ] **4. Create Unified Configuration** (2-3 hours)
  - Location: Create `config/training_config.yaml`
  - Action: Move all hardcoded parameters to config
  - Update: All 12 Python scripts to use config
  - Reference: AUDIT_REPORT.md Section 3.4, YAML template in AUDIT_SUMMARY.md

- [ ] **5. Fix Requirements.txt** (5 minutes)
  - Location: `requirements.txt`
  - Action: Add `pyyaml>=6.0`
  - Reference: AUDIT_REPORT.md Section 8.2

---

## High Value Improvements (Week 2)

- [ ] **6. Implement Data Deduplication** (1-2 hours)
  - Location: Create `scripts/phase3-prepare-data-mlx/05b_deduplicate_data.py`
  - Action: Find questions >85% similar, keep only most complete
  - Expected: Remove ~65 duplicates (5% of dataset)
  - Reference: AUDIT_REPORT.md Section 3.3

- [ ] **7. Add Data Quality Filtering** (2-3 hours)
  - Location: `scripts/phase3-prepare-data-mlx/05c_filter_by_quality.py`
  - Action: Score answers, keep top 80%
  - Criteria: Length, formatting, no truncation, no transcript references
  - Expected: Remove ~200 low-quality examples
  - Reference: AUDIT_REPORT.md Section 5.2

- [ ] **8. Add Reproducibility Tracking** (1 hour)
  - Location: Create `scripts/phase4-fine-tune-model/run_metadata.py`
  - Action: Record timestamp, git commit, parameters, dataset hash before training
  - Reference: AUDIT_REPORT.md Section 6.1

- [ ] **9. Create Troubleshooting Guide** (1-2 hours)
  - Location: Create `docs/TROUBLESHOOTING.md`
  - Sections: OOM errors, loss not decreasing, overfitting, data issues
  - Reference: AUDIT_REPORT.md Section 6.4

---

## Enhancement Phase (Week 3)

- [ ] **10. Implement Data Augmentation** (4-6 hours)
  - Location: Create `scripts/phase2-refine-raw-data/05_augment_qa_pairs.py`
  - Methods: Paraphrasing, length variants (short/medium/long)
  - Expected: 3x dataset expansion (1,367 → 4,000+)
  - Reference: AUDIT_REPORT.md Section 5.1

- [ ] **11. Add Dataset Analysis Script** (1 hour)
  - Location: Create `scripts/phase3-prepare-data-mlx/03_analyze_dataset.py`
  - Metrics: Length distributions, formatting rates, video coverage
  - Reference: AUDIT_REPORT.md Section 6.2

- [ ] **12. Add Structured Logging** (2-3 hours)
  - Location: Create `utils/logging_setup.py`
  - Action: Replace 200+ print statements with logging module
  - Reference: AUDIT_REPORT.md Section 4.2

- [ ] **13. Standardize Error Handling** (1-2 hours)
  - Location: Create `utils/exceptions.py`
  - Action: Define custom exception classes
  - Reference: AUDIT_REPORT.md Section 4.1

- [ ] **14. Add Type Hints** (2-3 hours)
  - Location: All 12 Python scripts
  - Action: Add parameter and return type annotations
  - Reference: AUDIT_REPORT.md Section 4.3

---

## Quick Wins (Can Do Anytime)

- [ ] Fix PyYAML import hardening
- [ ] Add connection pooling for API requests
- [ ] Simplify string matching in 03_generate_answers.py
- [ ] Add missing docstrings

---

## Issues by Severity

### CRITICAL (0 Current)
*(Note: Learning rate needs validation but not a code issue)*

### HIGH (3 issues)
1. No training metrics captured
2. No model evaluation script
3. Learning rate validation needed

### MEDIUM (4 issues)
1. Data not deduplicated
2. No unified configuration
3. Inconsistent error handling
4. No structured logging

### LOW (6+ issues)
1. Type hints missing
2. No reproducibility artifacts
3. No dataset analysis
4. No troubleshooting guide
5. No data augmentation
6. Connection pooling missing

---

## Estimated Effort Summary

| Phase | Items | Hours | Impact |
|-------|-------|-------|--------|
| Week 1 - Critical | 5 | 8-10 | HIGH |
| Week 2 - High Value | 4 | 6-8 | MEDIUM-HIGH |
| Week 3 - Enhancement | 5 | 10-15 | MEDIUM |
| **Total** | **14** | **24-33** | **A Grade** |

---

## Success Metrics

After implementing critical fixes, you should be able to:

- [ ] See training loss curves (train vs validation)
- [ ] Run evaluation on 5-10 test questions
- [ ] Verify model learned appropriate formatting
- [ ] Compare learning rates (1e-5 vs 5e-6)
- [ ] Reproduce exact training run with saved metadata
- [ ] Understand dataset composition (lengths, formatting, coverage)

---

## Notes

- Start with Week 1 items - they're prerequisite for optimization
- Week 2 items can run in parallel once Week 1 is complete
- Week 3 is optional but recommended for production deployment
- No breaking changes required - all improvements are additive
- Current training is safe to proceed; just add metrics capture

