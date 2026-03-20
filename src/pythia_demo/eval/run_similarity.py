"""Run static word similiarty eval for Pythia suite, over pre-training."""


import numpy as np
import os
import pandas as pd
import transformers
import torch


from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


import utils



### Models to test
MODELS = [
         'EleutherAI/pythia-14m',
         'EleutherAI/pythia-70m',
         'EleutherAI/pythia-160m',
         'EleutherAI/pythia-410m',
         'EleutherAI/pythia-1b',
         'EleutherAI/pythia-1.4b',
         'EleutherAI/pythia-2.8b',
         # 'EleutherAI/pythia-6.9b',
         # 'EleutherAI/pythia-12b',
          ]


STIMULI = "data/tasks/static_word_similarity/SimLex-999/SimLex-999.txt"





### Handle logic for a dataset/model
def main(df, mpath, revisions):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("number of checkpoints:", len(revisions))

    for checkpoint in tqdm(revisions):

        ### Set up save path, filename, etc.
        savepath = "data/outputs/pythia_demo/similarity/"
        if not os.path.exists(savepath): 
            os.mkdir(savepath)
        if "/" in mpath:
            filename = "simlex-distances_model-" + mpath.split("/")[1] + "-" + checkpoint +  ".csv"
        else:
            filename = "simlex-distances_model-" + mpath +  "-" + checkpoint + ".csv"

        print("Checking if we've already run this analysis...")
        print(filename)
        if os.path.exists(os.path.join(savepath,filename)):
            print("Already run this model for this checkpoint.")
            continue

        model = AutoModelForCausalLM.from_pretrained(
            mpath,
            # revision=checkpoint,
            # output_hidden_states = True,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(mpath, revision=checkpoint)


        n_layers = model.config.num_hidden_layers
        print("number of layers:", n_layers)
    
        n_params = utils.count_parameters(model)
    
        results = []

        ### TODO: Why tqdm not working here?
        for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

            ### Get word
            w1 = " {w}".format(w = row['word1'])
            w2 = " {w}".format(w = row['word2'])

            ### Run model for each sentence
            ### TODO: Double-check whether I should be adding space before static word?
            s1_outputs = utils.run_model(model, tokenizer, w1)
            s2_outputs = utils.run_model(model, tokenizer, w2)

            ### Now, for each layer...
            for layer in range(n_layers+1): # `range` is non-inclusive for the last value of interval
    
                ### Get embeddings for word
                s1 = utils.get_embedding(s1_outputs['hidden_states'], s1_outputs['tokens'], tokenizer, w1, layer)
                s2 = utils.get_embedding(s2_outputs['hidden_states'], s2_outputs['tokens'], tokenizer, w2, layer)
    
                ### Now calculate cosine distance 
                model_cosine = cosine(s1.cpu(), s2.cpu())
    

                ### Add to results dictionary
                results.append({
                    'word1': w1,
                    'word2': w2,
                    'Distance': model_cosine,
                    'Layer': layer,
                    'similarity': row['SimLex999']
                })
    
        df_results = pd.DataFrame(results)
        df_results['n_params'] = np.repeat(n_params,df_results.shape[0])
        df_results['mpath'] = mpath
        df_results['revision'] = checkpoint
        df_results['step'] = int(checkpoint.replace("step", ""))
        
        
    
        savepath = "data/outputs/pythia_demo/similarity/"
        if not os.path.exists(savepath): 
            os.mkdir(savepath)
    
        if "/" in mpath:
            filename = "simlex-distances_model-" + mpath.split("/")[1] + "-" + checkpoint +  ".csv"
        else:
            filename = "simlex-distances_model-" + mpath +  "-" + checkpoint + ".csv"
    
        df_results.to_csv(os.path.join(savepath,filename), index=False)


if __name__ == "__main__":

    ## Read stimuli
    df = pd.read_csv(STIMULI, sep = "\t")

    ### Get revisions
    revisions = utils.generate_revisions_test()

    ## Run main
    for mpath in MODELS:
        print("Running:", mpath)
        main(df, mpath, revisions)
