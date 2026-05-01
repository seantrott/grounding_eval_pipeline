"""
Alternative approach to Experiment 2 using log probability of TARGET WORD ONLY.

Instead of averaging over the full explicit sentence, we compute:
P(color_word | context + "The [object] was")

For example:
P("brown" | "John looked at the steak on his plate. The steak was")
vs
P("red" | "John looked at the steak on his plate. The steak was")

This is more direct and should be more sensitive to implicit color inference.
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
         'EleutherAI/pythia-6.9b',
         'EleutherAI/pythia-12b',
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


def next_seq_prob(model, tokenizer, seen, unseen):
    device = next(model.parameters()).device  # get model's actual device
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

        # Append next token to input
        next_token_tensor = torch.tensor([[unseen_id]], device=device)
        input_ids = torch.cat((input_ids, next_token_tensor), dim=1)

    total_log_prob = sum(log_probs)
    total_prob = torch.exp(total_log_prob)
    return total_prob.item()



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

        full_context = row['sentence_text'] + " " + row['explicit_text']
        target_word = " " + row['last_word']
        ablated_context = full_context.replace(target_word, "").rstrip('.').strip()

        prob = next_seq_prob(
            model = model,
            tokenizer = tokenizer,
            seen = ablated_context,
            unseen = target_word
        )
        log_probs.append(np.log2(prob))
    
    # Add to dataframe
    df['log_prob'] = log_probs
    df['model_name'] = model_name
    df['n_params'] = n_params
    
    # Save to CSV
    mpath = model_name.split("/")[1]
    output_path = f'data/outputs/pythia_demo/event_inference/log_probs2/{mpath}_final_logprob_results.csv'.format(mpath=mpath)
    df.to_csv(output_path, index=False)
    
    
    return df




if __name__ == "__main__":
    for model_name in MODELS:
        results = main(model_name)