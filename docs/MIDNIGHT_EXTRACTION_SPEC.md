# Frozen Midnight-12k feature preparation

This stage prepares a second pathology representation for the BreakHis improvement study. The public MIT-licensed kaiko-ai/midnight release is pinned atadc6b15679c981cce6f9b018bbad09d16eeeda9f. Its4,546,005,432-byte checkpoint is bound by SHA25652c14f20386ca17c2af8a7bf32c31c352668a8fbf6aefc88d86be6eaa0c72ca1. Pinned model configuration and model-card bytes are checked before execution. Existing Transformers4.51.3 and Torch2.6.0 load the official Dinov2 architecture locally with strict state matching.

Use the official RGB224resize-shorter-side,224center-crop,bilinear antialiased transform and channel mean/std0.5. Concatenate the final1,536-dimensional CLS vector with the mean final1,536-dimensional patch vector. The resulting3,072-dimensional field descriptors are frozen. BreakHis labels never enter encoder extraction. All7,909 images and four magnifications are represented with exact path coverage and checksums. Each cache records source,pretrained bytes,transform,environment and execution receipts. Pinned upstream notices remain in the private model cache.

First download and checksum the public files in a CPU SLURM job. A four-image fitting-free A40 technical extraction then checks strict loading,shape,finiteness and repeat output. Full extraction uses four magnification shards, two concurrent A40 GPUs,FP32,TF32disabled,batch8,workers0,30 minutes allocated per shard. This is two requested GPU hours for full extraction plus the separate10minute technical check. A40 nodes are currently available; the H100 node is drained at the scheduler check.

After all four caches pass coverage and byte checks, the planned classifier comparison uses the same original-eight-derived case roles and case-mean direct-five/dual-step logistic protocol with four C values. A field-supervised readout can be evaluated as its separately reported predeclared comparison. Its training configuration will bind the generated feature cache before any classifier fitting.

The official reported BreakHis benchmark selects TA,MC,FandDC at 40× and excludes lobular/papillary carcinoma. Its published score therefore supplies context for testing this representation. Current five-category performance still needs measurement. The pretraining source includes TCGA slides as recorded by the authors. All current fitted/evaluated images and labels come from BreakHis.

[Configuration](../configs/midnight-12k-v1.json) · [Primary-source audit](reviews/IMPROVEMENT_PRIMARY_RESEARCH.md).
