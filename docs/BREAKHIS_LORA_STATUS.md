# Midnight adapter comparison: execution status

November16,2026. The current verified best two-step accuracy remains **75.67%/75.00%** from the completed similarity classifier. The adapter study has no completed accuracy result yet. [Current result](BREAKHIS_KERNEL_RESULTS.md) · [Fixed adapter recipe](BREAKHIS_LORA_SPEC.md).

The full20-run comparison is submitted as array4980370, at most eight A40 GPUs concurrently. At the first check, eight tasks were running and twelve queued. Each original provider role has a direct-five control and two-step candidate, both using the same fixed20-epoch training recipe and four checkpoint opportunities. Collector4980371 is scheduled after successful completion of the complete array. No favorable partial result is promoted.

Immutable producer is `7daff480a44b35f57ca859390a08d447cdd9df46`. Configuration SHA256 is `d4559a4b9e559f422f06b5a6afcbdfd347eb72a1206456826df40671957488c5`. The input audit4980360 completed successfully in14seconds, binding all20 fixed-C1 initial checkpoints and original fitting-only normalizers.

Both fitting-only GPU technical checks4980364 completed successfully in66–67seconds each. They exercised head warmup and actual adapter gradients, preserved frozen original weights, and reproduced initial/reloaded fitting-probe features and probabilities. Independent CPU artifact review4980365 accepted both in23seconds, receipt SHA256 `45755af7a405b8fc0eb0bd85c8fe73d44e305124133b24074873fe4ef9cd4d4d`. Peak allocated GPU memory was about5.64GB. Only the official ToTensor deprecation warning appeared on stderr. [Actual technical review](reviews/LORA_SMOKE_REVIEW.md).

The full pinned local suite passed277 tests in68.695seconds, and source CI passed. These establish implementation behavior, not classification accuracy. The CPU collector will replay saved adapted field features and small states; it does not independently execute the raw-image adapted encoder. Producer reconstruction probes supply the image-execution consistency evidence.

The remaining sequence is: finish20 fits, verify all80 candidate checkpoints and selected fitting outputs, compare both arrangements with the accepted similarity and frozen-head references, then update the capstone report/presentation with all measured outcomes.94–95% final accuracy is still a goal, with no guarantee from these technical checks. No new dataset, split or threshold sweep is included.
