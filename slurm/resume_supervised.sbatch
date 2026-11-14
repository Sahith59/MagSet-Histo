#!/usr/bin/env bash
# Technical continuation only: preserve the interrupted artifacts before resume.
# The immutable training source, scientific settings and 20-epoch budget stay fixed.
#SBATCH --job-name=magset-supervised-resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
set -euo pipefail
: "${MAGSET_ROOT:?Set project root}"
: "${MAGSET_SOURCE:?Set original immutable training source}"
: "${MAGSET_CODE_SHA:?Set original training source SHA}"
: "${MAGSET_ORIGINAL_ARRAY:?Set interrupted array ID}"
[[ "$(cat "$MAGSET_SOURCE/.source-sha")" == "$MAGSET_CODE_SHA" ]]
[[ "$MAGSET_ORIGINAL_ARRAY" =~ ^[0-9]+$ ]]
[[ "$SLURM_ARRAY_TASK_ID" == 0 || "$SLURM_ARRAY_TASK_ID" == 1 ]]
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAGSET_SOURCE"
"$MAGSET_ROOT/envs/magset/bin/python" - <<'PY'
import json, os, re, shutil
from pathlib import Path
import torch
from magset.adapt_breakhis import bound_checkpoint
from magset.data import file_hash

torch.set_num_threads(2)
root=Path(os.environ['MAGSET_ROOT'])
gpu=torch.cuda.get_device_name(0)
required=os.environ.get('MAGSET_REQUIRED_GPU_FAMILY')
if required and required not in gpu:raise ValueError('Recovery GPU family differs')
trial=int(os.environ['SLURM_ARRAY_TASK_ID'])+1
folder=root/'runs/breakhis-supervised-v1/split812'/f'trial{trial}'
if (folder/'metrics.json').exists():raise ValueError('Completed model must not be rerun')
context=json.loads((folder/'started.json').read_text())
saved=bound_checkpoint(folder/'last.pt',context)
epoch=saved['epoch']
if not 1<=epoch<=20 or [h['epoch'] for h in saved['history']]!=list(range(1,epoch+1)):
    raise ValueError('Invalid completed-epoch boundary')
evidence=folder/('recovery-'+os.environ['SLURM_ARRAY_JOB_ID']+'-'+os.environ['SLURM_ARRAY_TASK_ID'])
evidence.mkdir()
paths=[folder/name for name in ('started.json','last.pt','last.tmp','selected.pt') if (folder/name).exists()]
paths+=sorted(folder.glob('selection-epoch*.npz'))
hashes={}
for path in paths:
    target=evidence/path.name
    shutil.copyfile(path,target)
    hashes[path.name]=file_hash(target)
    if file_hash(path)!=hashes[path.name]:raise ValueError('Interrupted-artifact copy differs')
receipt={'reason':'original scheduler wall-time limit; continue authoritative last.pt',
    'original_array':os.environ['MAGSET_ORIGINAL_ARRAY'],'recovery_array':os.environ['SLURM_ARRAY_JOB_ID'],
    'task':int(os.environ['SLURM_ARRAY_TASK_ID']),'seed':812,'trial':trial,
    'training_source_sha':os.environ['MAGSET_CODE_SHA'],'saved_epoch':epoch,'next_epoch':epoch+1,
    'scientific_epoch_limit':20,'selected_epoch_in_last':saved['best']['epoch'],
    'preserved_artifact_sha256':hashes,'recovery_node':os.environ.get('SLURMD_NODENAME'),
    'recovery_gpu':gpu,'orphan_files_are_not_resume_authority':True}
restore=os.environ.get('MAGSET_RESTORE_FROM_RECOVERY')
if restore:
    if not re.fullmatch('[0-9]+',restore):raise ValueError('Invalid recovery source ID')
    original=folder/('recovery-'+restore+'-'+os.environ['SLURM_ARRAY_TASK_ID'])
    prior=json.loads((original/'receipt.json').read_text())
    if (prior['training_source_sha']!=os.environ['MAGSET_CODE_SHA'] or prior['trial']!=trial
            or prior['original_array']!=os.environ['MAGSET_ORIGINAL_ARRAY']):
        raise ValueError('Restoration source binding differs')
    restored={}
    for name,digest in prior['preserved_artifact_sha256'].items():
        if name not in ('last.pt','selected.pt') and not re.fullmatch('selection-epoch[0-9]+[.]npz',name):continue
        source=original/name
        if file_hash(source)!=digest:raise ValueError('Original preserved checkpoint changed')
        stage=folder/(name+'.restore-tmp')
        shutil.copyfile(source,stage)
        if file_hash(stage)!=digest:raise ValueError('Checkpoint restoration copy differs')
        os.replace(stage,folder/name);restored[name]=digest
    saved=bound_checkpoint(folder/'last.pt',context);epoch=saved['epoch']
    if epoch!=prior['saved_epoch']:raise ValueError('Original epoch boundary differs')
    receipt.update(restored_from_recovery=restore,restored_artifact_sha256=restored,
                   saved_epoch=epoch,next_epoch=epoch+1,selected_epoch_in_last=saved['best']['epoch'])
(evidence/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({'resume_trial':trial,'saved_epoch':epoch,'next_epoch':epoch+1,'gpu':receipt['recovery_gpu']}),flush=True)
PY
export MAGSET_RESUME=1 MAGSET_SMOKE=0
bash "$MAGSET_SOURCE/slurm/breakhis_supervised.sbatch"
