# Controlled field and magnification fusion pilot

Registered before this round's cluster submission. This is an exploratory validation protocol, not the final test benchmark.

## Question and decision

The verified initial pilot has low recall for rare subtypes and some correct predictions become wrong after adding a view. Two hypotheses are testable: averaging field features may suppress useful image evidence, and fixed view weights may combine conflicting evidence poorly. Neither explanation is established by the error report alone.

The immediate decision is whether field preservation or learned fusion merits further investigation. Compare against the original linear baselines. Continue a complex mechanism only after showing a reproducible baseline weakness and a contribution beyond prior work. Success means a consistent validation improvement across trials and later split/training seeds, with rare-class recall and worst-subset results checked. A single best trial does not qualify.

## Fixed inputs and labels

Use the same provider-case manifest, pinned frozen encoder caches, five repeated holdouts, and seed-812 inner splits as pilot v1: 46 fitting, eight validation, 28 outer cases per trial. All four scales contain unpaired field bags. No image sequence is treated as a registered cross-scale pair. All fields are retained, with no outcome-driven field selection. Outer test scoring is unavailable in the new training CLIs.

Phikon-v2 is the first neural-screen encoder because it had higher descriptive pilot averages. This choice is exploratory and must be checked with a second encoder before a general claim. The field-linear controls use both existing encoders. No end-to-end backbone adaptation occurs in this round.

## Case-balanced normalization and field classifier

For each scale, average fitting-case field means and second moments with equal case weights. Derive channel standard deviations from those moments; replace values at or below 1e-6 by one. Validation and outer cases cannot fit these statistics. This differs from v1's normalization of already averaged case vectors, so raw score differences across rounds cannot isolate architecture alone.

The `field-late` control fits one logistic regression per scale to individual field embeddings. A field from case i with n fields and subtype c receives weight N_fit / (8 * N_cases_in_class_c * n). Thus field count does not multiply a case's influence and each class has equal total fitting weight. Use the same four C values as v1. Average field probabilities within case/scale and then across available scales. The paired `field-early-late` control uses exactly the same normalization but averages field features before fitting and classification. Both average available scale probabilities. This tests the fitting/aggregation strategy while holding normalization fixed; it does not by itself prove a particular biological explanation.

## Neural comparison matrix

Six methods cross field pooling (`mean`, standard gated-attention MIL) with view fusion (`mean`, contextual masked gate, one small self-attention block). These are established architectural controls, not proposed novelty.

Shared components: 128-dimensional projection, GELU, dropout 0.2, small learned scale embeddings, LayerNorm and an eight-output classifier. Field attention uses tanh/sigmoid gated scores within each scale. The view gate scores each view with the mean observed-view context. View attention has four heads and a 256-dimensional feedforward layer. Record parameter counts; these architectures are not automatically capacity matched.

Masks exclude padded fields before projection and exclude absent views from keys, values, pooling, and gates. Scale identity follows each token when reordered. At least one observed view and one field per observed view are required. Field/view permutations, absent contents (including NaNs), one-view inference, gradients, and checkpoint reload are tested.

## Equal selection budget within this matrix

Use all 15 nonempty subsets with equal weight in every fitting minibatch. Cases have equal weight within subtype; subtype weights are based only on the 46 fitting cases. Subset augmentation does not create new cases.

Train 100 fixed epochs, case batch size eight, AdamW, gradient norm cap one, float32, TF32 disabled, deterministic algorithms required. The four predeclared settings cross learning rate {0.0003, 0.001} and weight decay {0.01, 0.1}. Training seed 812 is fixed for this screen. **Validate only at the final epoch**, once per setting. Select equal-subset mean eight-class macro F1; ties retain the earlier grid entry. No epoch search, early stopping, or outer-test tuning.

The neural and linear methods have four parameter settings each; their training procedures and parameter capacities differ. The six neural methods share exactly the same selection opportunities. Report all methods, including regressions. Follow with split-seed and training-seed sensitivity before selecting a final mechanism.

## Outputs and execution

Save normalization, selected model and optimizer states, RNG states, source/config/cache hashes, all four candidate results, training losses, 120 validation probability records, elapsed time, GPU memory, and parameter count. Reload the selected checkpoint and verify probability equality. Independently recompute metrics from saved probabilities using the collector.

Twenty CPU field-linear jobs (two encoders × two aggregation policies × five trials) run with at most four workers. A separate real-data GPU smoke checks all six neural paths at two epochs; it is execution evidence only. Its success gates 30 neural configurations, at most four GPUs concurrently. Each neural configuration includes the four settings. Use new output directories; preserve previous runs and stop on incomplete/invalid results.

## Required next checks

- Use the paired normalization-matched field controls to interpret aggregation changes.
- Repeat the screen under prespecified alternate inner splits and training seeds; report instability.
- Reconcile strict identifier grouping and add the image-level benchmark track before final reporting.
- Audit close methods and secure a compatible second dataset before claiming a broadly novel mechanism.

## Prespecified stability follow-up

After the first complete matrix showed no neural improvement over the initial mean baseline, fix the follow-up before launching it: all six neural methods × five trials at split seed 813/training seed 812; the mean/mean and attention/attention controls × five trials at split seed 812/training seed 813; and the original Phikon mean linear control × five trials at split seed 813. Keep the original four-setting grids and fixed final epochs. Save each split/training-seed combination in a distinct matrix and do not merge different seeds into one five-trial ranking. Report unfavorable outcomes as well as favorable ones. A second split/training seed diagnoses sensitivity; it does not resolve limited case counts or justify independent-fold confidence intervals.
