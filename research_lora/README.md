# Research LoRA

This is a self-contained experiment package. It never imports or edits `research_v2`, and it does not alter any `baseline` source file. The original DiffPlanner is loaded from an explicit `args.json` plus full baseline checkpoint; only the adapter checkpoint is written.

## Remote-server workflow

On the remote server, start from `/home/lisw/programs/Nuplan-Diffusion-Baseline` and use absolute paths for data and outputs. Build a clean manifest first:

```bash
cd /home/lisw/programs/Nuplan-Diffusion-Baseline
python -m research_lora.scripts.build_manifests --style-index /data/style_index.jsonl --output-dir /exp/lora/manifests
python -m research_lora.selftest
python -m research_lora.scripts.train_style_lora --config research_lora/configs/smoke.yaml --args-file /baseline/args.json --baseline-checkpoint /baseline/model.pth --manifest /exp/lora/manifests/train.jsonl --val-manifest /exp/lora/manifests/val.jsonl --cache-root /data/cache --style aggressive --steps 1000 --output /exp/lora/checkpoints/aggressive_lora.pt
```

Train conservative with the same command and `--style conservative`. Each checkpoint contains *only its declared branch*; aggressive and conservative must be supplied together for signed-rho evaluation. Checkpoints store LoRA weights, insertion layers, hashes of the baseline/normalizer, manifest hash, configuration and validation statistics. A hash mismatch is rejected during loading.

`evaluate_denoising` is the mandatory inexpensive gate before `evaluate_open_loop`. It loads a separately constructed untouched DiffPlanner for the rho-zero identity comparison. The open-loop script computes free-drive/car-follow **generated-trajectory proxy axes** directly from generated trajectories, compares each signed rho with expert target distributions, and reports cluster hit rate, per-axis Wasserstein and MMD per scene. These are proxy-space results; they must not be described as membership in the source index's original clustering space. It uses the baseline decoder's SDE and DPM-Solver through the transparent wrapper and scans the prescribed rho grid with paired seeds.

First-stage injection is MLP plus final output projection only. `attention_lora.py` exists for a later, self-attention-only ablation; it is deliberately excluded from the default injector.
