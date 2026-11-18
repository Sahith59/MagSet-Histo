# Why the original dual-step UNI2-h misses rare cases

On the fixed 17-case validation set, the original network **recognized both lobular cases as malignant in all 30 zoom conditions**. In 25 answers it named ductal cancer. For the one papillary case, it called ten conditions benign and five mucinous. Even when we supplied the correct malignant decision, its second step chose mucinous in all 15 conditions. The previously reported eight benign and seven mucinous answers belong to the revised subtype-initialization experiment. [Original score traces](evidence/CASEFIRST_RARE_DECISIONS.json) · [Revised experiment](BREAKHIS_SUBTYPE_INITIALIZATION.md)

| Development data | True type | Cases / distinct groups | Correct exact diagnosis, all zoom combinations | Correct if the first step is given the true malignant answer | Correct using all four zooms together |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed validation | Lobular | 2 / 1 | 5/30 | 5/30 | 0/2 |
| Fixed validation | Papillary | 1 / 1 | 0/15 | 0/15 | 0/1 |
| Training cases, each group held out once | Lobular | 2 / 2 | 18/30 | 19/30 | 1/2 |
| Training cases, each group held out once | Papillary | 4 / 4 | 1/60 | 16/60 | 0/4 |

With 40× alone, both validation lobular cases received the correct label. Combining all four zooms led to a ductal call in both cases. Every one of the five correct lobular answers included 40×. On the four held-out papillary training cases the network gave the right final answer once, with 40× alone. An entire held-out papillary case was called benign under all 15 combinations, while the subtype scores favored papillary under all 15. Other papillary cases continued to have subtype errors even when we supplied their malignant label. These case-specific patterns do not establish a preferred zoom. [Mask-by-mask counts](evidence/CASEFIRST_RARE_DECISIONS.json)

### What the image descriptions tell us

Using the inputs immediately before the learned classifier, we compared the visual description of each held-out case to fitting cases under the same available zooms. For each true lobular case we compared its mean cosine similarity to the fitting lobular cases against its mean similarity to fitting ductal cases. For papillary cases we compared papillary against mucinous. Each fitting case contributed equally within its class. All normalization used only the cases available for fitting that fold.

- The validation papillary case was closer on average to the fitting mucinous cases in **15/15** conditions. This occurred in **33/60** papillary training-group holdout conditions as well.
- Both validation lobular cases were closer on average to the fitting lobular cases in **30/30** conditions, despite the 25 ductal calls. The two held-out training lobular cases also showed higher average similarity to their own type in **30/30** conditions.

The papillary result points to a problem already visible in the image descriptions for this case. The lobular result shows the fitted decision can disagree with this simple measure of resemblance. The distances compare two named classes in each analysis. They do not measure five-category accuracy or determine whether a pathologist could see the diagnostic cues. [Reproducible calculation](../scripts/audit_casefirst_rare_decisions.py)

### What to examine next

The private blinded review packet contains **84 original fields and 84 center crops** from 21 development cases. Its review sheet and the working copy had zero recorded observations at this audit. A qualified reviewer can compare the full field with the displayed crop and record tissue cues visible at the edges and at the model input resolution. A paired wider-image experiment can measure whether adding that tissue changes recognition. Use the same cases, available zoom combinations and decision rules. Report both decision stages so an improvement in one cannot hide a decline in the other.

This analysis checked the original validation score archive against its recorded hash and all 42 training fold archives against the previously published inventory commitment. It used saved logits and frozen descriptions on development roles. No fitting, external cases or additional 16-case benchmark predictions were involved. The 15 answers for each case share its tissue and reference diagnosis. [Original validation replay](evidence/FIELD_HIERARCHY_REPLAY.json) · [Training-group replay](evidence/FIELD_HIERARCHY_GROUP_CHECK.json)
