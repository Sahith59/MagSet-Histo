# Capstone prediction demonstration

## Open the recorded viewer

Open [CAPSTONE_PREDICTION_REPLAY.html](evidence/CAPSTONE_PREDICTION_REPLAY.html) in a browser. It works offline. Tick any of40×/100×/200×/400× to see the corresponding real saved outputs. All15 nonempty combinations are available; no selected view shows a helpful empty state.

This is a **viewer of verified saved predictions**. It does not run the encoder, accept uploaded images, or estimate clinical accuracy. The original-image computation was performed separately on the cluster.

The retained trial1/arrangement812 model processed121 raw fields (29/30/32/30 by magnification). Fresh job4971403 completed in39seconds with no fitting or outer-role scoring. All15 outputs reproduced archived validation probabilities within6.05e-08. The fixed lexicographic validation-case rule predates its outcome; the example was not selected by searching for success. The full-view example is adenosis with56.1% subtype model score and81.3% derived benign score. These are model scores for one example, not accuracy or calibrated clinical risk.

Saved head SHA256: `318a38488b84845b0731be38fe7d8d55e46739526f0ef196a4ba480152db8f21`. Prediction source: `9448a7e319167b7417670a882552edb372603fd3`. The wrapper was introduced at `b8498cad8665f7c8818990d8bc13f3d0d6c7bf4e`; updating a viewer does not alter that inference source.

## Actual model execution

Use the existing configured cluster environment, original dataset, pinned pretrained encoder and saved trial1 head. Those artifacts remain private and are not bundled in Git. The scheduler entry is [capstone_replay.sbatch](../slurm/capstone_replay.sbatch). It checks the immutable source marker, exact model/protocol hashes, validation roles and archived predictions. It refuses an existing output path. There is no new model selection or fitting.

On the configured cluster, after loading the project-root/account/partition environment:

```bash
export MAGSET_CODE_SHA=9448a7e319167b7417670a882552edb372603fd3
export MAGSET_SOURCE="$MAGSET_ROOT/sources/$MAGSET_CODE_SHA"
export MAGSET_DEMO_OUTPUT="$MAGSET_ROOT/runs/breakhis-milestone-v1/capstone-replay-NEW.json"
sbatch --account="$MAGSET_ACCOUNT" --partition="$MAGSET_GPU_PARTITION" \
  --export=ALL --output="$MAGSET_ROOT/logs/capstone-replay-%j.out" \
  --error="$MAGSET_ROOT/logs/capstone-replay-%j.err" \
  "$MAGSET_ROOT/repo/slurm/capstone_replay.sbatch"
```

Choose a new output name first. This command is provided for reproducibility; the fresh checked replay already exists, so no extra GPU job is needed for the presentation.

## What to say while showing it

“These are microscope pictures from one validation case. The fixed picture reader turns each picture into numbers. We average within each zoom, combine available zooms and ask a small classifier for eight subtype scores. Here I switch between recorded view combinations. Adding the four malignant subtype scores gives the separate benign/malignant answer. A successful example shows that inference works; our overall accuracy comes from the complete validation tables.”

[Full results](CAPSTONE_BEFORE_AND_NOW.md) · [Secondary binary rules and limits](CAPSTONE_BINARY_RESULTS.md) · [Delivery plan](CAPSTONE_DELIVERY_PLAN.md).
