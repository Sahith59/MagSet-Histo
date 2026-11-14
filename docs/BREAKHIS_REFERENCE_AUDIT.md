# Coverage and published-reference compatibility audit

November13,2026. Read-only audit after the saved-model diagnosis. No author code was imported/executed, no model fitted, no outer predictions, and no case roles changed.

## Our release and ten existing splits

All7,909 manifest paths, filenames and recorded image hashes are unique. All82 subtype-qualified provider cases have all four views. Independent reconstruction matches the ten trial/seed splits and20 saved frozen/adapted memberships/hashes. No provider-case, image-path or recorded-hash role overlap was found.

| Subtype | Fitting cases | Validation cases | Excluded outer-role cases |
| --- | ---: | ---: | ---: |
| Adenosis |1|1|2|
| Ductal carcinoma |25|1|12|
| Fibroadenoma |6|1|3|
| Lobular carcinoma |2|1|2|
| Mucinous carcinoma |5|1|3|
| Papillary carcinoma |3|1|2|
| Phyllodes tumor |1|1|1|
| Tubular adenoma |3|1|3|
| Total |46|8|28|

The one-case fitting limitation of adenosis/phyllodes is a property of this fixed protocol, not missing rows from our parser. Numeric-stem/bare-ID overlaps still prevent a verified clinical-patient-disjointness claim. These remain provider-case exploratory splits. [Independent coverage review](reviews/BREAKHIS_COVERAGE_REVIEW.md).

## Pinned public reference inspected

Author repository [breakhis-graph-transformer](https://github.com/zaiedhussein/breakhis-graph-transformer), commit `09371ccff9860cb4ef9cb1a6625f6fd269e8da64`; downloaded archive SHA256 `692ba1d803fb49c3f97670bbd944372dae0b6e2dad7a351b2f4fd11bd5f28d60`. Source remains private locally; this report contains aggregate findings and links.

Current [configuration](https://github.com/zaiedhussein/breakhis-graph-transformer/blob/09371ccff9860cb4ef9cb1a6625f6fd269e8da64/src/breakhis_graph/config/schema.py) and [runner](https://github.com/zaiedhussein/breakhis-graph-transformer/blob/09371ccff9860cb4ef9cb1a6625f6fd269e8da64/src/breakhis_graph/training/experiment_runner.py) specify a pretrained Swin-based model, image-level supervised training, a separate outer-training calibration subset for checkpoint selection/temperature fitting, and per-magnification case aggregation. The default ActIII config uses20 epochs. Its focal loss, augmentation, image-class balancing, calibration and checkpoint selection differ from our frozen linear reference.

Independent standard-library inspection checked all80 [released split CSVs](https://github.com/zaiedhussein/breakhis-graph-transformer/tree/09371ccff9860cb4ef9cb1a6625f6fd269e8da64/resources/locked_splits) against our exact manifest filenames. Every image/label/magnification maps correctly; each train/validation pair covers its magnification and is disjoint within that pair. However:

- Released multiclass folds contain65 fitting-side and17 validation-side provider cases, before the runner's separate calibration carve-out. Our experiment uses46 fitting/eight validation cases. These are materially different protocols.
- The released multiclass CSVs have85 provider-case validation appearances but82 distinct cases: three repeat across folds. The current strict generator's disjoint-fold description therefore does not exactly describe these artifacts.
- One bare identifier contains two different subtype-qualified cases. The current parser returns qualified IDs, but released CSVs retain bare IDs. The case-aggregation helper groups by that bare key and takes the first label. In multiclass fold2 validation, this can merge ductal/lobular cases if those CSV IDs reach aggregation unchanged. All eight image subtypes are actually present; treating it as a missing image class would be incorrect.
- In multiclass fold3, two adenosis cases switch train/validation roles between40×/100× and200×/400×. Combining all four released per-view folds would put both cases in fitting and validation. They must not be directly unioned for our joint four-view task. Binary roles agree across views.

These are release/code identity and compatibility findings. They do not establish which exact historical pipeline produced the paper's headline numbers or prove that the publication is invalid. They do prevent an honest claim that copying these CSVs reproduces our task or provides a clean joint four-view split. Keep author-reported values separate from new implementation results.

## Decision

Preserve our exact roles and all15-subset endpoint. Establish a clearly labeled standard supervised image-model reference, rather than claim an exact graph-paper reproduction. The registered [ResNet50 pipeline](BREAKHIS_SUPERVISED_SPEC.md) uses regularization and retains its trained image head for case aggregation. It changes a complete pipeline and cannot isolate a novel fusion contribution. Its settings and fit budget are fixed before outcomes; independent source review and a fitting-only GPU smoke gate production.

No clean new confirmation cohort is created by this audit. TCGA and novelty work remain deferred.
