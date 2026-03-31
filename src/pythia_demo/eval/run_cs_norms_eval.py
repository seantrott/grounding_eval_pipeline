"""
Evaluate Pythia models' ability to predict contextualized sensorimotor norms.

This script:
1. Extracts contextualized embeddings for target words from Pythia
2. Trains regression models (RidgeCV) to predict sensorimotor dimensions
3. Evaluates using cross-validation (k-fold and leave-one-word-out)

Key design choices:
- Single forward pass per sentence, caching all layers' embeddings
- Exact sliding-window token matching (primary), with fallback logging
- StandardScaler applied within each CV fold
- RidgeCV for automatic regularization tuning
"""

import logging
import numpy as np
import os
import pandas as pd
import torch

from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Logging setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("sensorimotor_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================
MODELS = [
    'EleutherAI/pythia-14m',
    'EleutherAI/pythia-70m',
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    # 'EleutherAI/pythia-1b',
    # 'EleutherAI/pythia-1.4b',
    # 'EleutherAI/pythia-2.8b',
]

STIMULI = "data/tasks/property_inference/cs_norms/cs_norms_with_string.csv"

# Target columns (sensorimotor dimensions with .M suffix = means)
SENSORIMOTOR_COLS = [
    'Vision.M', 'Hearing.M', 'Olfaction.M', 'Interoception.M', 'Taste.M', 'Touch.M',
    'Foot_leg.M', 'Mouth_throat.M', 'Torso.M', 'Head.M', 'Hand_arm.M'
]

# RidgeCV alpha search space
RIDGE_ALPHAS = np.logspace(-2, 4, 20)


# ============================================================
# Token matching
# ============================================================

def find_target_token_indices(tokens, tokenizer, target_word):
    """
    Find token indices for a target word using exact sliding-window matching.
    
    Strategy:
    1. Tokenize the target word directly and do a sliding-window search
       over the full token sequence.
    2. If that fails, try without leading space.
    3. If still no match, return None (caller decides what to do).
    
    Returns:
        list of int indices, or None if not found.
    """
    tokens_list = tokens.tolist()
    
    # Try with the target as given (usually " word" with leading space)
    for variant in [target_word, target_word.strip()]:
        target_ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(target_ids) == 0:
            continue
        
        # Sliding window: find exact subsequence match
        for i in range(len(tokens_list) - len(target_ids) + 1):
            if tokens_list[i:i + len(target_ids)] == target_ids:
                return list(range(i, i + len(target_ids)))
    
    return None


def get_embedding_from_cache(all_hidden_states, layer, target_indices):
    """
    Extract embedding for target word from cached hidden states at a given layer.
    
    Args:
        all_hidden_states: tuple of (batch, seq, hidden) per layer
        layer: which layer to use (0 = embedding layer)
        target_indices: list of token positions to average over
    
    Returns:
        numpy array of shape (hidden_dim,)
    """
    layer_hidden = all_hidden_states[layer][0]  # (seq, hidden)
    target_embeddings = layer_hidden[target_indices]  # (n_tokens, hidden)
    return target_embeddings.mean(dim=0).cpu().numpy()


# ============================================================
# Embedding extraction (single forward pass, cache all layers)
# ============================================================

def extract_all_layer_embeddings(df, model, tokenizer, n_layers):
    """
    Run one forward pass per sentence and cache embeddings for ALL layers.
    
    Returns:
        embeddings: dict mapping layer -> numpy array of shape (n_samples, hidden_dim)
        match_failures: list of (index, word, sentence) tuples that failed matching
    """
    n_samples = len(df)
    hidden_dim = model.config.hidden_size
    total_layers = n_layers + 1  # include embedding layer (layer 0)
    
    # Pre-allocate arrays for all layers
    embeddings = {layer: np.zeros((n_samples, hidden_dim)) for layer in range(total_layers)}
    match_failures = []
    
    for row_idx, (df_idx, row) in enumerate(tqdm(df.iterrows(), total=n_samples, desc="Extracting embeddings")):
        sentence = row['sentence']
        word = row['string']
        target = f" {word}"
        
        # Single forward pass
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        tokens = inputs['input_ids'][0]
        hidden_states = outputs.hidden_states  # tuple of (1, seq, hidden) x (n_layers+1)
        
        # Find target token indices (once per sentence, shared across layers)
        target_indices = find_target_token_indices(tokens, tokenizer, target)
        
        if target_indices is None:
            match_failures.append((df_idx, word, sentence))
            # Fallback: use last non-padding token
            target_indices = [len(tokens) - 1]
            logger.warning(
                f"Token match failed for word='{word}' in sentence='{sentence[:80]}...' "
                f"(using last token as fallback)"
            )
        
        # Extract embedding at every layer
        for layer in range(total_layers):
            embeddings[layer][row_idx] = get_embedding_from_cache(hidden_states, layer, target_indices)
    
    if match_failures:
        logger.warning(f"Total token match failures: {len(match_failures)}/{n_samples}")
    else:
        logger.info("All token matches succeeded.")
    
    return embeddings, match_failures


# ============================================================
# Evaluation functions
# ============================================================

def make_ridge_pipeline():
    """Create a StandardScaler + RidgeCV pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=RIDGE_ALPHAS))
    ])


def evaluate_no_cv(X, y, sensorimotor_cols, model_name="Model"):
    """Fit on all data (sanity check — will overfit)."""
    pipe = make_ridge_pipeline()
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    
    results = {}
    for i, feat in enumerate(sensorimotor_cols):
        r, p = pearsonr(y[:, i], y_pred[:, i])
        results[feat] = {'r': r, 'r2': r**2, 'p': p}
    
    r_overall, _ = pearsonr(y.flatten(), y_pred.flatten())
    results['overall'] = {'r': r_overall, 'r2': r_overall**2}
    
    logger.info(f"{model_name} (no CV): overall r={r_overall:.3f}, r²={r_overall**2:.3f}")
    return results, y_pred


def evaluate_kfold(X, y, sensorimotor_cols, model_name="Model", n_splits=10):
    """K-fold CV with scaling inside each fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y)
    
    for dim in tqdm(range(y.shape[1]), desc=f"{model_name} (KFold)", leave=False):
        pipe = make_ridge_pipeline()
        y_pred[:, dim] = cross_val_predict(pipe, X, y[:, dim], cv=kf, n_jobs=-1)
    
    results = {}
    for i, feat in enumerate(sensorimotor_cols):
        r, p = pearsonr(y[:, i], y_pred[:, i])
        results[feat] = {'r': r, 'r2': r**2, 'p': p}
    
    r_overall, _ = pearsonr(y.flatten(), y_pred.flatten())
    results['overall'] = {'r': r_overall, 'r2': r_overall**2}
    
    logger.info(f"{model_name} ({n_splits}-fold): overall r={r_overall:.3f}, r²={r_overall**2:.3f}")
    return results, y_pred


def evaluate_logo(X, y, groups, sensorimotor_cols, model_name="Model"):
    """Leave-one-group-out CV (strictest test — no word leakage across folds)."""
    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)
    
    for dim in tqdm(range(y.shape[1]), desc=f"{model_name} (LOGO)", leave=False):
        pipe = make_ridge_pipeline()
        y_pred[:, dim] = cross_val_predict(pipe, X, y[:, dim], cv=logo, groups=groups, n_jobs=-1)
    
    results = {}
    for i, feat in enumerate(sensorimotor_cols):
        r, p = pearsonr(y[:, i], y_pred[:, i])
        results[feat] = {'r': r, 'r2': r**2, 'p': p}
    
    r_overall, _ = pearsonr(y.flatten(), y_pred.flatten())
    results['overall'] = {'r': r_overall, 'r2': r_overall**2}
    
    logger.info(f"{model_name} (LOGO): overall r={r_overall:.3f}, r²={r_overall**2:.3f}")
    return results, y_pred


# ============================================================
# Main
# ============================================================

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(df, mpath):
    """Main evaluation for a single model (final checkpoint)."""
    groups = df['word'].values
    
    savepath = "data/outputs/pythia_demo/property_inference/cs_norms/"
    os.makedirs(savepath, exist_ok=True)
    
    model_name = mpath.split("/")[-1] if "/" in mpath else mpath
    filename = f"sensorimotor_predictions_{model_name}.csv"
    
    if os.path.exists(os.path.join(savepath, filename)):
        logger.info(f"Already processed: {filename}")
        return
    
    # Load model
    logger.info(f"Loading model: {mpath}")
    model = AutoModelForCausalLM.from_pretrained(
        mpath, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(mpath)
    
    n_layers = model.config.num_hidden_layers
    n_params = count_parameters(model)
    hidden_dim = model.config.hidden_size
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {n_layers}, Hidden dim: {hidden_dim}, Params: {n_params:,}")
    
    y = df[SENSORIMOTOR_COLS].values
    
    # ---- Single pass: extract embeddings for ALL layers ----
    layer_embeddings, match_failures = extract_all_layer_embeddings(df, model, tokenizer, n_layers)
    
    # Save match failures for auditing
    if match_failures:
        fail_df = pd.DataFrame(match_failures, columns=['index', 'word', 'sentence'])
        fail_path = os.path.join(savepath, f"match_failures_{model_name}.csv")
        fail_df.to_csv(fail_path, index=False)
        logger.info(f"Saved {len(match_failures)} match failures to {fail_path}")
    
    # ---- Evaluate each layer using cached embeddings ----
    all_results = []
    
    for layer in range(n_layers + 1):
        logger.info(f"Evaluating layer {layer}/{n_layers}")
        
        X = layer_embeddings[layer]
        
        results_nocv, _ = evaluate_no_cv(X, y, SENSORIMOTOR_COLS, f"L{layer}")
        results_kf, _ = evaluate_kfold(X, y, SENSORIMOTOR_COLS, f"L{layer}")
        results_logo, _ = evaluate_logo(X, y, groups, SENSORIMOTOR_COLS, f"L{layer}")
        
        layer_result = {
            'model': model_name,
            'n_params': n_params,
            'layer': layer,
            'r2_nocv': results_nocv['overall']['r2'],
            'r2_kfold': results_kf['overall']['r2'],
            'r2_logo': results_logo['overall']['r2'],
            'r_nocv': results_nocv['overall']['r'],
            'r_kfold': results_kf['overall']['r'],
            'r_logo': results_logo['overall']['r'],
        }
        
        for feat in SENSORIMOTOR_COLS:
            layer_result[f'{feat}_r_logo'] = results_logo[feat]['r']
            layer_result[f'{feat}_r2_logo'] = results_logo[feat]['r2']
        
        all_results.append(layer_result)
    
    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(savepath, filename), index=False)
    logger.info(f"Saved: {filename}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (overall r² by layer)")
    print("=" * 60)
    print(f"{'Layer':<10} {'No CV':<10} {'10-Fold':<10} {'LOGO':<10}")
    print("-" * 40)
    for res in all_results:
        print(f"{res['layer']:<10} {res['r2_nocv']:<10.3f} {res['r2_kfold']:<10.3f} {res['r2_logo']:<10.3f}")
    
    # Clean up
    del model
    del layer_embeddings
    torch.cuda.empty_cache()


if __name__ == "__main__":
    df = pd.read_csv(STIMULI)
    df = df[df['Class'] == 'N']
    
    logger.info(f"Loaded {len(df)} noun stimuli")
    logger.info(f"Unique words: {df['word'].nunique()}")
    logger.info(f"Sensorimotor dimensions: {SENSORIMOTOR_COLS}")
    
    for mpath in MODELS:
        main(df, mpath)