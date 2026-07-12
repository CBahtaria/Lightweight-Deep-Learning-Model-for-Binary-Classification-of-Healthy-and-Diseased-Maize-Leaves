# maize-leaf-classifier — CLAUDE.md

Lightweight deep-learning binary classifier (healthy vs diseased maize leaves). Target deployment: smallholder-farmer inference on low-bandwidth, low-compute edge devices. Audience: agricultural extension workers in SADC.

Architectures evaluated: InceptionV3, MobileNetV2, ResNet50, VGG16, Xception. MobileNetV2 is the production candidate for edge deployment due to parameter count and inference latency.

---

## Blocking Gates (must be satisfied before any commit to main)

**Phase 1 — Secret scan. Never proceed past this step until it passes.**
```bash
gitleaks detect --source . --no-git
```
`gitleaks detect` must return 0 findings. No API keys, HuggingFace tokens, or cloud storage credentials in source.

**Phase 2 — Notebook output scrub.**
All Jupyter notebooks committed to main must have outputs stripped (no embedded training logs, no weight tensors, no base64 image previews):
```bash
jupyter nbconvert --clear-output --inplace "Training Codes/"*.ipynb
```

**Phase 3 — Model accuracy regression.**
Any new training run must match or exceed the last recorded baseline accuracy in `Results/`. Record the comparison before committing a new weights file.

---

## Model Routing

- **Sonnet 4.6**: implementation, notebook review, training script edits, data pipeline changes.
- **Opus 4.7**: architecture selection, deployment strategy, cross-architecture comparison analysis.
- **Haiku 4.5**: quick file reads, grep, existence checks.

---

## Review Rubric

An agent's work is **APPROVED** only if ALL of the following hold:

| # | Criterion | How to verify |
|---|-----------|---------------|
| a | Every file the agent claims to have touched has a visible diff | `git diff --name-only` matches the agent's report |
| b | Every finding has a fix AND a verification step | Finding IDs map 1-to-1 in the agent record below |
| c | No new secrets committed | `gitleaks detect` → 0 findings |
| d | Every test the agent named actually ran and passed | Test output included in agent record, no skips |
| e | No doc claim contradicts source | Cross-check any prose claim against the actual file before marking done |

---

## Agent Record Schema

```yaml
agent:         # agent ID or session label
model:         # opus | sonnet | haiku
phase:         # phase number or name within the task
status:        # DONE | NEEDS_REVIEW | BLOCKED
files_touched:
  - Training Codes/MobilenetV2 Training.ipynb
findings:
  - id:          F-001
    severity:    high | medium | low | info
    title:       Short description
    fix:         What was changed
    verified_by: Command or manual step that confirmed the fix
tests_run:
  - name: accuracy regression check
    command: compare Results/ baseline vs new run
tests_result: PASS | FAIL | SKIP
residual_risk: >
  Any known limitation or deferred item before production edge deployment.
```

---

## Key Constraints

- No model weights (`.h5`, `.pt`, `.onnx`, `.tflite`) committed to git unless they are the final production artifact and fit under 50 MB. Large weights go to object storage (document the URL in `Results/`).
- Notebooks in `Training Codes/` must run top-to-bottom without manual intervention on a clean Colab/Kaggle environment.
- Dataset preprocessing must be documented; do not assume the dataset path is stable across environments — parameterise it.
- Any new architecture evaluation must be recorded in `Results/` with: model name, parameter count, top-1 accuracy, inference latency (ms), and hardware used.
