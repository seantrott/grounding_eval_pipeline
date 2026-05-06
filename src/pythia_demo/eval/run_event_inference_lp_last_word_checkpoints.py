"""
Compute log probability of the TARGET WORD ONLY across Pythia model checkpoints.

For each trial, we compute:
    P(target_word | ablated_context)
e.g.,
    P("brown" | "John looked at the steak on his plate. The steak was")

Iterates through training checkpoints (steps) for each Pythia model so we can
track how implicit color inference (and similar event-inference effects) emerge
during pretraining.

NOTE: We process one item at a time (no batching) to avoid any padding-related
artifacts in surprisal/probability values.

Usage:
    python get_logprobs_event_inference.py --model EleutherAI/pythia-1.4b
    python get_logprobs_event_inference.py --model EleutherAI/pythia-2.8b --checkpoints 0 1000 143000
    python get_logprobs_event_inference.py --run_all
"""

import argparse
import os
import time
import torch
import pandas as pd
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


ALL_CHECKPOINTS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000
]

ALL_MODELS = [
    "EleutherAI/pythia-14m",
    # "EleutherAI/pythia-70m",
    # "EleutherAI/pythia-160m",
    # "EleutherAI/pythia-410m",
    # "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-1.4b",
    # "EleutherAI/pythia-2.8b",
    # "EleutherAI/pythia-6.9b",
    # "EleutherAI/pythia-12b",
]


def count_parameters(model):
    """Count trainable parameters."""
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        total_params += parameter.numel()
    return total_params


def next_seq_prob(model, tokenizer, seen: str, unseen: str) -> float:
    """
    Compute P(unseen | seen) by sequentially feeding tokens of `unseen`
    after `seen` and multiplying token probabilities.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(seen, return_tensors="pt").to(device)
    unseen_ids = tokenizer.encode(unseen)

    log_probs = []
    for unseen_id in unseen_ids:
        with torch.no_grad():
            logits = model(input_ids).logits
        next_token_logits = logits[0, -1]
        next_token_probs = torch.softmax(next_token_logits, dim=0)
        prob = next_token_probs[unseen_id]
        log_probs.append(torch.log(prob))

        next_token_tensor = torch.tensor([[unseen_id]], device=device)
        input_ids = torch.cat((input_ids, next_token_tensor), dim=1)

    total_log_prob = sum(log_probs)
    total_prob = torch.exp(total_log_prob)
    return total_prob.item()


def run_checkpoint(model_name: str, revision: str, tokenizer, df: pd.DataFrame,
                   device: str, output_dir: str):
    """Load a specific checkpoint and compute log probs for all items."""

    checkpoint_str = f"step{revision}"
    mpath = model_name.split("/")[-1]
    output_path = os.path.join(output_dir, f"{mpath}_{checkpoint_str}_logprobs.csv")

    # Skip if already computed
    if os.path.exists(output_path):
        print(f"  Skipping {checkpoint_str} — already exists at {output_path}")
        return

    print(f"  Loading checkpoint: {checkpoint_str}")
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name, revision=checkpoint_str, use_safetensors=False
    ).to(device)
    model.eval()

    n_params = count_parameters(model)

    log_probs = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {checkpoint_str}", leave=False):
        full_context = row['sentence_text'] + " " + row['explicit_text']
        target_word = " " + row['last_word']
        ablated_context = full_context.replace(target_word, "").rstrip('.').strip()

        prob = next_seq_prob(
            model=model,
            tokenizer=tokenizer,
            seen=ablated_context,
            unseen=target_word,
        )
        log_probs.append(np.log2(prob))

    out_df = df.copy().reset_index(drop=True)
    out_df['log_prob'] = log_probs
    out_df['model'] = model_name
    out_df['checkpoint'] = int(revision)
    out_df['n_params'] = n_params

    out_df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    # Free memory
    del model
    torch.cuda.empty_cache()


def run_single_model(model_name, checkpoints, input_path, output_dir, device):
    """Run all checkpoints for a single model."""
    print(f"\nLoading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} trials")
    print(f"Running {len(checkpoints)} checkpoints for {model_name}")

    model_start = time.time()

    for ckpt in checkpoints:
        print(f"\n--- Checkpoint {ckpt} ---")
        run_checkpoint(
            model_name=model_name,
            revision=str(ckpt),
            tokenizer=tokenizer,
            df=df,
            device=device,
            output_dir=output_dir,
        )

    elapsed = time.time() - model_start
    print(f"\n{model_name} done in {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-14m")
    parser.add_argument("--input", type=str,
                        default="data/tasks/event_inference/all_items_tidy.csv")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=None,
                        help="Specific checkpoints to run. Defaults to all.")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all Pythia models (14m through 12b).")
    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = "data/outputs/pythia_demo/event_inference/log_probs_checkpoints/"
    os.makedirs(args.output_dir, exist_ok=True)

    # Checkpoints to evaluate
    checkpoints = args.checkpoints if args.checkpoints else ALL_CHECKPOINTS

    if args.run_all:
        total_start = time.time()
        print(f"\n{'='*50}")
        print(f"Running all {len(ALL_MODELS)} Pythia models")
        print(f"Checkpoints per model: {len(checkpoints)}")
        print(f"{'='*50}")

        for i, model_name in enumerate(ALL_MODELS):
            print(f"\n{'='*50}")
            print(f"[{i+1}/{len(ALL_MODELS)}] {model_name}")
            print(f"{'='*50}")
            run_single_model(model_name, checkpoints, args.input, args.output_dir, device)

        total_elapsed = time.time() - total_start
        print(f"\n{'='*50}")
        print(f"ALL DONE in {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
        print(f"Results in {args.output_dir}")
        print(f"{'='*50}")
        print(f"\nREMEMBER: Terminate your instance!")

    else:
        run_single_model(args.model, checkpoints, args.input, args.output_dir, device)

    print(f"\nDone! All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()