# Running on the Mac Studio fleet

Three Mac Ultra 512 GB (supermac01-03) and five Mac Ultra 256 GB (bigmac01-05), reachable by their Tailscale MagicDNS names (bigmac01 ... supermac03; the SSH config's HostName is
the bare name), linked by Thunderbolt with RDMA. Access is by the dedicated key `~/.ssh/sichos_fleet_ed25519`
(SSH config block "sichos-fleet" on this Mac; `cluster/hosts.txt` lists the hosts, coordinator first).

Per-node copy model (no shared volume needed): `push.sh` rsyncs the pipeline, day records,
audio and cached CT2 models into `~/Sichos` on every node; nodes work on their own copies;
`pull_days.sh` merges results back (each shard writes disjoint day files). A shared SMB/NFS
volume mounted at one path everywhere also works with ROOT pointed at it.

```
cluster/hosts.txt            the eight hosts, coordinator first
cluster/check_fleet.sh       key-based SSH test + specs for every host
cluster/setup_node.sh        per node: venv with requirements-align.txt (run via ssh host 'bash -s' < ...)
cluster/push.sh              rsync data + pipeline + models to all nodes in parallel
cluster/align_fleet.sh       mode 1: alignment fan-out (WORKERS processes per node)
cluster/progress.sh          aligned-day count per node
cluster/pull_days.sh         merge day records back; prints coverage
cluster/setup_train_node.sh  training env without admin rights: uv-managed Python 3.12 + requirements.txt (done on supermac01-03)
cluster/build_and_scan.sh    after alignment: every node rebuilds the identical dataset and scans its train shard
cluster/pull_scan.sh         pull the shard reports and write train.clean.jsonl here
cluster/train_fleet.sh       mode 2: one training experiment per node (uses .venv-train)
cluster/train_dp_rounds.sh   mode 3: one model on all nodes by periodic averaging
```

## Mode 1: alignment fan-out (no new code path)

`align_day.py --shard i/n` already partitions the day list. `align_fleet.sh` starts
`WORKERS` (default 6) processes per node, each with 4 CPU threads for CTranslate2 int8,
writing `sync_local` into disjoint day JSONs on that node; `pull_days.sh` merges them back.

Sequence: `check_fleet.sh` -> `setup_node.sh` on each host -> `push.sh` -> `align_fleet.sh` ->
`progress.sh` until done -> `pull_days.sh` -> `build_dataset.py` on this Mac -> `build_and_scan.sh`
(nodes rebuild the same dataset and scan the train split in 40 shards) -> `pull_scan.sh`.
Measured: 5 workers x 4 threads per node, ~170-215 s per 11-minute day per worker.

Measured on this M3 Pro: 2-3x realtime per 4-thread process. A 24-P-core Ultra with 6
processes should give about 15x realtime per node if memory bandwidth does not bind.

| nodes | anchor model | 90 h (517 days, 5786-87) | ~220 h (~1,270 days, with 5783-85) |
|---:|---|---:|---:|
| 8 | turbo (validated) | ~45 min | ~2 h (1,372 days) |
| 3 (the 512s) | turbo | ~2 h | ~5 h |
| 8 | large-v3 (more anchors) | ~2.5 h | ~6 h |

Then on the coordinator: `build_dataset.py --year 5786 5787 --dev-days ... --test-days ...`
and `compute_offsets.py` already done here. Hold out about 20 days each for dev and test.

## Mode 2: one experiment per node (tonight, zero risk)

`train_fleet.sh hosts.txt train_configs.txt` assigns line i of the configs file to node i.
The example matrix covers large-v3 and turbo, full fine-tune at two learning rates, LoRA,
and frozen-encoder variants. Each node writes `runs/<name>/`; score each with
`evaluate.py` on the same test manifest and keep the best. This uses the fleet fully
without any inter-node communication, and the experiments are the ones you would want to
run anyway. PyTorch on MPS, fp32.

## Mode 3: one model on all nodes by periodic averaging

`train_dp_rounds.sh hosts.txt ROUNDS`: every round each node trains `STEPS` (default 200)
optimizer steps on its own 1/N shard of the clips from the previous round's averaged
weights (constant learning rate), then the coordinator averages the N checkpoints
(`average_checkpoints.py`), converts to CTranslate2 and scores dev. Communication is a
checkpoint write/read per round on the share (6 GB per node for large-v3, seconds over
Thunderbolt), so RDMA is not needed. Scaling is near-linear in throughput; the quality
cost of averaging every 200 steps instead of every step is usually small, and the dev score
after each round shows it. Effective batch per round = N x STEPS x batch x accum clips.

## Mode 4 (not built): synchronous data parallel over RDMA

For the fastest wall clock, an MLX training loop with `mx.distributed` (its backend for
Thunderbolt RDMA on macOS 26) all-reducing gradients every step. PyTorch on MPS cannot
use the RDMA link. Estimated build: one to two days, including porting the model
forward/backward to MLX and validating against the PyTorch trainer on the pilot set.
Worth it if you will train many models; not needed for the first full model.

## Measured on the fleet (2026-09-17)

- Alignment: 1,372 days in 3 h 17 min with 8 nodes x 5 workers x 4 threads (turbo anchors);
  172-285 s per 10-minute day per worker under full load. Zero node failures.
- supermac01 training throughput, large-v3 full fine-tune, MPS fp32, no checkpointing (while also
  aligning): 0.304 clips/s at batch 8, 0.421 clips/s at batch 16. Full five-year set is ~48k clips
  per epoch: ~32 h/epoch on one Ultra, ~4-5 h/epoch on eight (mode 3). Frozen-encoder turbo ~10x less.

## Training time estimates (from measured throughput)

Measured on this M3 Pro (18 GPU cores, PyTorch MPS, fp32, batch 2, gradient checkpointing,
6-step runs including warm-up, so conservative):

| run | clips/s on M3 Pro |
|---|---:|
| turbo, decoder-only (pilot, batch 4, no checkpointing) | 1.3 |
| turbo, full fine-tune | 0.158 |
| large-v3, full fine-tune | 0.061 |

Extrapolation to one Ultra: x4.5 for GPU cores (80 vs 18) and x1.5 for steady state with
batch 8-16 and no checkpointing (memory is not a constraint there), about x6-7 overall.
Epoch = ~13,500 clips (~106 h paired audio after filtering). Mode 3 adds ~10% for rounds.

| run, 3 epochs | one Ultra | 8 Ultras (mode 3) |
|---|---:|---:|
| turbo, frozen encoder | ~2 h | ~15 min |
| turbo, full fine-tune | ~11 h | ~1.5 h |
| large-v3, full fine-tune | ~28 h | ~4 h |
| large-v3, LoRA r=64 | ~20 h | ~3 h |

A native MLX trainer (mode 4) would likely be 2-4x faster again. For comparison, one H100
does the large-v3 full fine-tune in about 2-4 h, so eight Ultras in mode 3 match it and in
mode 4 beat it. Verify on the fleet with one node: `finetune.py --max-steps 30` prints
`train_samples_per_second`.

## Memory rules for training on the Macs (learned the hard way)

- Always launch through `train_fleet.sh`: it sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` and `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6`, refuses a
  node with less than `MIN_FREE_GB` (80) unused, and appends `EXIT <code>` to the run log.
- 256 GB nodes: full and LoRA fine-tunes need `--gradient-checkpointing` with `--batch 8 --grad-accum 4`.
  Frozen-encoder runs (~50 GB turbo, ~120 GB large-v3) fit as they are. 512 GB nodes ran large-v3 full
  (266 GB) and large-v3 LoRA (359 GB) without checkpointing.
- Check 5 minutes after launch: step `1/` visible, a `[mem] step 1` line, no `EXIT` line.

## GB10 on the network

On the tailnet as `biggb01` (Linux). The fleet key is not installed there yet; install
`~/.ssh/sichos_fleet_ed25519.pub` in that user's `authorized_keys` and set the SSH user in the config.

One DGX Spark-class box (GB10, 128 GB unified, Arm CPU). Treat it as the primary trainer for the
large-v3 full fine-tune (CUDA bf16; likely 3-6 h for 3 epochs, calibrate first) while the Macs run
the config matrix and all CT2 inference (alignment, scanning, transcription). If CTranslate2 has no
CUDA build for aarch64, score on it with `evaluate.py --backend hf`. See AGENT_HANDOFF.md 6c.

## Recommended sequence

1. `setup_node.sh` on all nodes (about 15 min, mostly downloads to the shared cache).
2. Mode 1 alignment on all 8 nodes (under an hour). Build the dataset. Scan the train split
   with the base model on 8 nodes with `evaluate.py --shard`-style day lists if needed, then
   `filter_manifest.py`.
3. Mode 2 overnight: the eight configs. Frozen-encoder runs finish in hours and give a
   usable model the same night; full fine-tunes finish the next day.
4. Mode 3 or 4 for the final large-v3 model once the best recipe is known.

## Added 2026-09-21: re-scoring, idle-node lists, GPU transcription

- `hosts_idle.txt` (supermac01 bigmac01 bigmac04 bigmac05) = nodes free while the v2 wave trains on the other four;
  `hosts_v3.txt` + `train_configs_v3.txt` = the self-training wave (pseudo-labels from the Phase 7 transcripts).
- `rescore_node.sh <ct2-dir|hf-id> <report-name> [suppress]` runs ON a node: dev+test clips in 5 shards, then the
  20 long-form test days in 5 shards, with a decoding option (`allow-quotes` frees the gershayim, see
  `pipeline/decoding.py`). Reports `<name>-shardN[-longform]/` on the node; pull them into `asr/reports/aq/`.
- `transcribe_fleet.sh` now syncs `pipeline/` to every node before starting workers, takes `EXTRA_ARGS`
  (default `--suppress allow-quotes` for the CT2 backend) and `BACKEND=hf HF_WORKERS=2` to run the HF fp32 model on
  each node's GPU instead (`pipeline/hf_longform.py`: VAD windows <= 28 s decoded without timestamps, ~11-20x
  realtime per process, better long-form WER than faster-whisper for these models). With `BACKEND=hf`, MODEL is the
  HF checkpoint dir (must exist on the origin node; the script copies it to the others).
- Measured 2026-09-21: 2 HF workers per node + a concurrent HF eval on the same GPU still gave ~11x realtime each.

## Disk rule (learned 2026-09-23)

A run directory holds up to three `checkpoint-*` folders (about 9 GB each for turbo, 17 GB for large-v3, optimizer
state included) until `harvest_node.sh` deletes them. Never stage a run directory to another node with those folders:
use `rsync -az --exclude 'checkpoint-*'` (transcribe_fleet.sh does), touch the three harvest markers on the copy, and
check `df` on the receiving node first. supermac01 filled up this way and the run training there died at its next
checkpoint save; it was resumed with `finetune.py --resume` after the half-written checkpoint folder was removed.
