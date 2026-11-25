#!/bin/bash
set -euo pipefail
cd "$HOME/bpe-xray"
echo "=== [1/6] Plain LoRA extra (Compound + Adversarial BPE) ==="
./job.sh src/eval_plain_extra.py

echo "=== [2/6] BPE-V2 Word Retina Frozen (n=128) ==="
./job.sh src/train_v2.py --model Qwen/Qwen3-0.6B-Base --arm v2_frozen --n_train 128 --n_test 60 --epochs 5

echo "=== [3/6] BPE-V2 Word Retina Frozen (n=512) ==="
./job.sh src/train_v2.py --model Qwen/Qwen3-0.6B-Base --arm v2_frozen --n_train 512 --n_test 60 --epochs 5

echo "=== [4/6] BPE-V2 Word Retina Frozen (n=2048) ==="
./job.sh src/train_v2.py --model Qwen/Qwen3-0.6B-Base --arm v2_frozen --n_train 2048 --n_test 60 --epochs 5

echo "=== [5/6] BPE-V2 Word Retina Frozen (n=4096, save ckpt) ==="
./job.sh src/train_v2.py --model Qwen/Qwen3-0.6B-Base --arm v2_frozen --n_train 4096 --n_test 60 --epochs 5 --save_ckpt

echo "=== [6/6] BPE-V2 Word Retina + LoRA (n=4096, save ckpt) ==="
./job.sh src/train_v2.py --model Qwen/Qwen3-0.6B-Base --arm v2_lora --n_train 4096 --n_test 60 --epochs 4 --save_ckpt

echo "=== ALL V2 EXPERIMENTS COMPLETE ==="
