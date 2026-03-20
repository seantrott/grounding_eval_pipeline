"""
Replication of Experiment 2 from Jones, Bergen, & Trott (2024):
"Do Multimodal Large Language Models and Humans Ground Language Similarly?"
Computational Linguistics, 50(4): 1415-1440

This script uses CONDITIONAL LOG PROBABILITIES instead of representation similarity.

EXPERIMENT DESIGN:
For each item, we compute:
P(explicit_text | sentence_text)

If the model is sensitive to implied features, matching pairs should have
higher log probability (less negative).

OUTPUT:
Adds log_prob column to the tidy dataframe
"""

import pandas as pd
import numpy as np
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings
import utils
warnings.filterwarnings('ignore')


### Models to test
MODELS = [
         # 'EleutherAI/pythia-14m',
         # 'EleutherAI/pythia-70m',
         # 'EleutherAI/pythia-160m',
         # 'EleutherAI/pythia-410m',
         # 'EleutherAI/pythia-1b',
         'EleutherAI/pythia-1.4b',
         'EleutherAI/pythia-2.8b',
         # 'EleutherAI/pythia-6.9b',
         # 'EleutherAI/pythia-12b',
          ]


def get_model_and_tokenizer(model_name):
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoXForCausalLM.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    return model, tokenizer


def get_conditional_log_prob(context, continuation, model, tokenizer):
    """
    Compute log probability of continuation given context.
    
    Args:
        context: Context sentence (e.g., "John looked at the steak on his plate")
        continuation: Continuation to score (e.g., "The steak was brown.")
        model: Language model
        tokenizer: Tokenizer
    
    Returns:
        Average log probability per token in continuation
    """
    # Combine context and continuation
    full_text = context + " " + continuation
    
    # Tokenize
    context_tokens = tokenizer(context, return_tensors="pt", add_special_tokens=True)
    full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
    
    context_len = context_tokens['input_ids'].shape[1]
    full_len = full_tokens['input_ids'].shape[1]
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(full_tokens['input_ids'])
        logits = outputs.logits  # (batch, seq_len, vocab_size)
    
    # Calculate log probabilities for continuation tokens only
    log_probs = []
    
    for i in range(context_len, full_len):
        # Get predicted distribution at position i-1 (predicting token i)
        token_logits = logits[0, i-1, :]
        log_probs_dist = torch.log_softmax(token_logits, dim=0)
        
        # Get actual token at position i
        actual_token = full_tokens['input_ids'][0, i]
        
        # Get log prob of actual token
        log_prob = log_probs_dist[actual_token].item()
        log_probs.append(log_prob)
    
    # Return average log probability per token
    if len(log_probs) > 0:
        return np.mean(log_probs)
    else:
        return 0.0


def main(model_name):
    # Load tidy data
    print("Loading data...")
    df = pd.read_csv('data/tasks/event_inference/all_items_tidy.csv')
    print(f"Loaded {len(df)} rows from tidy file\n")
    
    # Load model
    model, tokenizer = get_model_and_tokenizer(model_name)
    
    n_params = utils.count_parameters(model)
    
    # Compute log probs for each row
    log_probs = []
    
    print("Computing log probs...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rows"):
        log_prob = get_conditional_log_prob(
            row['sentence_text'], 
            row['explicit_text'], 
            model, 
            tokenizer
        )
        log_probs.append(log_prob)
    
    # Add to dataframe
    df['log_prob'] = log_probs
    df['model_name'] = model_name
    df['n_params'] = n_params
    
    # Save to CSV
    mpath = model_name.split("/")[1]
    output_path = f'data/outputs/pythia_demo/event_inference/log_probs/{mpath}_all_logprob_results.csv'.format(mpath=mpath)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Total rows: {len(df)}")
    
    print("\nMatch effect summary:")
    summary = df.groupby(['dataset', 'match'])['log_prob'].agg(['mean', 'std'])
    print(summary)
    
    return df

if __name__ == "__main__":
    for model_name in MODELS:
        results = main(model_name)