# Master's capstone delivery plan

Updated November14,2026 from the user's confirmed timing. **Prepare the complete package by November17–18. Presentation is expected around November20; the exact day is not fixed.** Professor Dong has not specified the prediction task. Keep the existing eight-subtype study and report the source-defined benign/malignant secondary outcome separately.

## Deliverables and dates

| Working date | Deliverable | Acceptance evidence |
| --- | --- | --- |
| Nov14 | Freeze the evidence and model | Three independent GPT-6.1 Sol reviews; original outputs recovered; all tested pipelines retained; regional comparison complete; frozen Phikon reference retained |
| Nov15 | Finish the prediction demonstration and report tables | Recorded original-image inference replay plus clearly labeled binary score; command for rerunning the saved model; both validation arrangements and all trials retained |
| Nov16 | Build presentation draft and complete report | Problem, data, architecture, split/metric definitions, historical/current distinction, both label tasks, negative experiments, limitations and references |
| Nov17 | Complete reviewable capstone package | Slide deck, report, runnable code/instructions, result ledger, demo and a practice explanation |
| Nov18 | Revision/rehearsal buffer | Correct labels/numbers; rehearse timing and common questions; incorporate any actual advisor requirements |
| Around Nov20 | Presentation | Date/format to be confirmed; no automatic message or submission |

The dates are a work plan, not a promise of95% accuracy or institutional acceptance. Model search is frozen for this package. A change requires a specific capstone requirement or a concrete correctness defect, not a desire to obtain a larger number.

## Presentation contract

- **Historical supplied notebooks:**93.73%/94.99% saved40× image-random reused validation. They were not independently retrained and are not new-patient test results.
- **Eight-subtype case study:** retain the frozen reference and report its modest accuracy/F1 openly. Do not claim that the completed adaptation, supervised ResNet or regional-view experiments improved it when their gates failed.
- **Benign/malignant secondary output:** fixed sum of the dataset's malignant subtype probabilities, fixed0.5 threshold, no new fitting. Report all three accepted earlier pipelines, both arrangements and all15 view subsets. State that the analysis is retrospective and does not replace the eight-class primary endpoint.
- **Prediction demonstration:** distinguish actual saved-model/raw-image replay from an interactive display of its saved outputs. A viewer is not a live image-upload model. A correct demonstration is execution evidence, not aggregate accuracy.
- **Scope:** prototype/reproducibility/generalization study for the master's capstone. A novel mechanism, fresh clinical confirmation and conference-paper claims remain later work.

## What must be in the report

Use [the before/now explanation](CAPSTONE_BEFORE_AND_NOW.md), [project-report draft](../paper/breakhis-project-report.md), [image endpoint bridge](CAPSTONE_BRIDGE_RESULTS.md) and fixed binary secondary analysis. Include dataset source/release,7,909 images/four magnifications/eight labels,82 provider keys versus unresolved clinical identity,46/8/28 roles, overlapping trials,31 distinct validation cases, exact inference chain, both successful execution checks and negative model outcomes.

Never present training accuracy as held-out accuracy, AUROC as accuracy, binary accuracy as eight-subtype accuracy, or dependent subset/image/trial events as independent patients. No fresh confirmation cohort is created by resplitting these reused cases.

## Likely questions to practise

1. What goes into the model, and what does each output mean?
2. Why are the notebook95% and current subtype scores different?
3. Why can benign/malignant accuracy exceed exact-subtype accuracy?
4. How are case identities separated, and what uncertainty remains?
5. Why do more images/GPUs not solve the one-case rare-class limitation?
6. What did the failed experiments teach us, and what would a future study need?

Do not claim that Professor Dong has approved this scope or that a presentation/submission has already occurred.
