# Improving the original dual-step UNI2-h classifier

The capstone permits one network to learn a benign/malignant decision and a four-way malignant subtype decision together. The earlier case-first UNI2-h architecture meets this requirement. Its highest observed five-category accuracy on the reused 16-case benchmark was 199/240 (82.92%) with 231/240 (96.25%) binary. The later single five-probability output was a separate architecture experiment with 144/240 (60.00%) and 190/240 (79.17%). The decision to resume the earlier architecture came after viewing the benchmark. Any improvement now needs development evidence and eventual evaluation on new cases. [Matched scorecard](BREAKHIS_MODEL_COMPARISON.md)

## Experiment fixed before fitting

On the original 17-case validation set, the earlier model correctly named ductal cancer in 98/105 answers, lobular in 5/30 and papillary in 0/15. There were only two lobular and four papillary cases in training. The single change tested here gives each malignant type a recognizable starting pattern in the same existing network. This is a hypothesis about fitting with few examples. It adds no cases or model parameters.

For each fitting fold, form the mean normalized case descriptor for each malignant type, giving each case and its 15 zoom combinations equal weight within that type. Initialize eight of the 32 existing hidden units in opposite-direction pairs and their connections to the four subtype scores so each initial score equals cosine similarity to its type's mean. Keep the original binary output weights at initialization and train all classifier weights jointly. Changing shared hidden units can also change binary predictions. All other settings follow the earlier case-first model, including frozen UNI2-h features, normalization inside each fitting fold, seed 812, balanced losses, ten AdamW updates, threshold and diagnosis rule. Compare against an ordinary re-fit of the original network on the same folds. [Original fitting configuration](../configs/breakhis-shared-hierarchy-v1.json)

Run a paired leave-one-group-out check within the 49 training cases (42 groups), then fit on all 49 and evaluate both models once on the same 17 validation cases. Keep both outcomes regardless of direction. Progress requires more correct five-category answers with binary accuracy and mean per-zoom-subset macro F1 holding or rising in both checks. Verify the control against the earlier aggregate confusion matrices and F1, and independently replay every saved model's decisions. The existing 16-case follow-up stays closed to the candidate. [Training and verification code](../scripts/run_subtype_initialization.py)

## First result

| Development check | Original five | Revised five | Original binary | Revised binary | Original / revised macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Each of 42 training groups held out once | 484/735 (65.85%) | 501/735 (68.16%) | 673/735 (91.56%) | 694/735 (94.42%) | .4492 / .4668 |
| Same 17 validation cases | 193/255 (75.69%) | 202/255 (79.22%) | 229/255 (89.80%) | 232/255 (90.98%) | .4982 / .5493 |

The candidate passed the development rule fixed above. Three validation cases gained correct zoom answers, one lost answers and thirteen had the same correct count. On validation the subtype gains were five more ductal answers (98 to 103 of 105) and four more lobular (5 to 9 of 30). The sole papillary case still had zero correct answers. With all four zooms available together the exact diagnosis improved from 13/17 to 14/17. In the 42-group training check, benign and ductal answers improved and mucinous fell by two. The network trained on all 49 cases with seed 812 has been frozen with checkpoint SHA-256 `32759ae69fa48bea4eb354a6b009fd0da3fdd27e1087dafc740600576fb419d1`. It has no 16-case score.

Slurm job 4990990 completed in 32 seconds. Both arms and every fold replayed from their saved weights. A separate comparison matched every control decision to the historical archives and recomputed both accuracies and macro F1 from the prediction files. An initial scheduler attempt ended in zero seconds before training. [Aggregate counts, confusions and experiment revisions](evidence/SUBTYPE_INITIALIZATION.json)

## Training-seed check fixed after the first result

Repeat the 42-group training comparison with the next three integer seeds, 813, 814 and 815. Preserve every outcome and keep the original seed-812 checkpoint fixed. This checks how much the effect depends on the random starting weights and training dropout. It reuses the same 49 training cases. Neither the 17 validation cases nor the 16 benchmark cases receive predictions. The stability rule requires a five-category gain in at least two of three seeds and positive mean gain in five-category accuracy, with mean binary accuracy and macro F1 holding or improving across the three seeds. Report results even if this check fails.
