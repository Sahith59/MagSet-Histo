# Can one five-category output make both decisions?

## Registered comparison

Use only BreakHis. The 49 fitting cases and 17 validation cases, the 15 nonempty combinations of the four microscope magnifications, and the frozen UNI2-h center features match the previous reference. The previously used 16-case follow-up stays outside the run. Each validation case contributes 15 answers. The 17 cases belong to 15 conservative groups.

Fit one linear layer that gives five scores for benign, ductal, lobular, mucinous and papillary. Train it with one five-category loss. Each class has equal total fitting weight to give the two lobular and four papillary training cases a voice. At prediction, convert the five scores to fractions summing to one. If the four malignant fractions sum to at least one half, return the largest malignant score. Otherwise return benign. This fixed arithmetic makes both answers from a single output. The old reference learned a binary score and four subtype scores with different loss terms.

Use seed 812, AdamW, learning rate 0.001, weight decay 0.01 and 40 full-batch updates. At updates 10, 20, 30 and 40 calculate macro F1 separately for each zoom condition, then average. Keep the first update with the highest value. Score five-category and binary accuracy at that same update. The strict targets require at least 230 and 243 correct answers out of 255 respectively. Report class counts, both stages' errors and all-four-view performance. Recompute scores independently from the weights and pinned inputs. Comparison with the original changes classifier capacity, training loss and routing together. It evaluates the complete design. [Registered settings](../configs/breakhis-single-output-center-v1.json).

## Field review

Use the fixed anonymous packet of 84 fields from 21 training and validation cases. Compare the original field with the exact crop that reached UNI2-h. Record geometry and a simple color proxy separately from the qualified reader's observations. A supplemental feature check will run the same frozen reader on left, center and right windows at the same pixel scale and measure how similar their outputs are. This check uses no tumor labels or classifier fitting. An expert's observations are needed to tell whether excluded tissue carries cues that distinguish malignant subtypes. The next decision about wider coverage depends on those observations.
