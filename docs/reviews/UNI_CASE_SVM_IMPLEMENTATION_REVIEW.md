# UNI2-h case-SVC implementation review

Reviewed before real case-SVC fitting: the registered configuration, core math, producer, independent collector, scheduler wrapper and their synthetic tests. **No substantive correctness blocker was found.** This review establishes implementation consistency, not model performance or attainment of the >90% final-five / >97% binary objectives. The registration remains a bounded, standard linear-versus-RBF readout comparison on reused development data.

The reviewed configuration SHA256 is `66fc988ce6452a30552eb3191885238b88b4df901d28a24a3007f2e52b2d4bbc`. Source SHA256 values are:

| File | SHA256 |
| --- | --- |
| `magset/uni_case_svm.py` | `da6f305c5f0c69b64135c077b4507660b7dc9aa2e51741ee088ded803c0b7ff8` |
| `scripts/run_uni_case_svm.py` | `0bb0891ee2273c655906ae7152d1338999435776b86ba388c1bb8ff4a8c66941` |
| `scripts/collect_uni_case_svm.py` | `6cc92be2bd5e1331ba493a69bb22f4586dacf01c9cd5eeb497187dacde24ee05` |

## Data, descriptors and weights

The producer validates the original audit and exact cache bytes, checks a canonical path bijection and provider/label/group/view alignment, then averages original frozen field features within each case and magnification in float64. Cache row position is not used as the case identity. Each query case's mean uses only its own fields. The shared per-view, per-coordinate mean and population standard deviation are fitted on exactly the 49 TRAIN case means; neither validation nor test contributes to these statistics. The malignant head uses this same TRAIN-only transform, as registered.

For each of the 15 fixed nonempty subsets, the code averages only active transformed views, normalizes the 1536-vector by `max(norm, 1e-12)`, then appends four availability flags scaled by 0.5. It does not normalize again after appending flags. Case-major mask ordering agrees across fitting targets, sample weights, archives and records. The collector separately reconstructs all means, scales and descriptors from the pinned frozen cache and requires exact saved preprocessing arrays.

Binary fitting uses 49 cases × 15 masks; malignant fitting uses only the 35 malignant TRAIN cases × 15 masks. A single factor of `1/15` supplies one total sample mass per case. `class_weight='balanced'` is applied once by SVC, giving effective total mass 49/2 per binary class and 35/4 per malignant class. The collector independently recomputes targets, class weights, sample weights, descriptor hashes, support ownership and dual coefficient bounds. Resolved `gamma='scale'` uses the respective head's unweighted fitting matrix and is recorded even for the linear kernel.

## Margins and routing

Both arms use `sklearn.svm.SVC`; the linear control is not `LinearSVC`. Four shared C settings produce eight candidates and 16 head estimators. The four-class malignant estimator internally fits six pairwise problems. The collector reconstructs each pair from public support vectors, class support offsets, dual coefficients and intercepts. Its OVR transform assigns a pairwise vote to class i for a nonnegative i-versus-j margin, accumulates signed confidence, then adds `confidence / (3 * (abs(confidence) + 1))` to the votes. The public binary coefficients already use sklearn's binary sign convention. These sign conventions agree with the reviewed producer's public decision-function calls.

The authoritative router is binary margin ≥ 0, with zero routed malignant. The subtype is the first maximum of the four malignant OVR scores. The code never substitutes a five-class joint-score argmax. Sigmoid and softmax create explicitly uncalibrated normalized reporting scores; `probability=False` avoids probability-fitting cross-validation. A tiny negative margin can round to sigmoid 0.5. The registered guard deliberately rejects that disagreement instead of moving a threshold or nudging a saved score; both producer and collector enforce it.

## Selection and completion

All eight validation archives are generated after model serialization and validation reload checks. Selection uses each arm's first maximum mean of the 15 fixed-five-label macro F1 values. Both selections, every candidate artifact, preprocessing and context are hash-bound in the common family gate before the first selected test prediction. Only two selected retrospective archives are produced. The independent collector verifies those bindings and reconstructs all eight validation and both selected test outputs without fitting or importing the new producer's math.

The RBF progression screen compares the independently selected arms on validation: macro F1 must improve by at least .02 and accuracy must not decrease. The already viewed 16-case follow-up cannot override that decision. Full cache arrays and historical reference summaries are loaded into memory; the narrower, supported claim is that TRAIN determines fitting/statistics and validation determines selection, with selected test model calls after the gate.

## Evidence and limits

This reviewer ran six additional synthetic algebra checks with **zero estimator fits, zero private feature queries and zero jobs**: held-out perturbation invariance of normalization, case/mask layout and flags, inactive-view invariance, six-pair OVR arithmetic including ties, scalar versus matrix linear/RBF kernels, and the exact-zero/tiny-negative routing guard. The OVR helper comparison used the locally installed sklearn 1.5.2 implementation; production separately requires pinned sklearn 1.6.1. The existing 21 synthetic tests were read; their successful execution was reported by the parent reviewer, not repeated here under the no-fitting instruction.

One receipt limitation remains: the collector verifies the required source, preprocessing, gate, candidate and selected-test artifacts directly but does not require the redundant outer producer `result.json` inventory. Thus its accepted outcome would establish completeness of those individually checked artifacts, not verification of that outer inventory or an independent recount of runtime fit calls. This is not an identified arithmetic or selection defect.

No real case-SVC outcome is asserted in this review. It does not independently replay raw UNI2-h images or biological labels, certify optimizer optimality from scratch, or turn reused cases and dependent mask events into fresh confirmation. The private machine-check receipt is hash-bound with SHA256 `cafd8daa766e8a9d20f2e209b426a517bcc97fb690e0738529080ad934f470c6`.
