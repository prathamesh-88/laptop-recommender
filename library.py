
import numpy as np 
import pandas as pd 
import spacy


def extract_best_indices(m, topk, mask=None):
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) 
    best_index = index[mask][:topk]  
    return best_index


def predict_spacy(model, query_sentence, embed_mat, topk=3):

    query_embed = model(query_sentence)
    mat = np.array([query_embed.similarity(line) for line in embed_mat])
    # keep if vector has a norm
    mat_mask = np.array(
        [True if line.vector_norm else False for line in embed_mat])
    best_index = extract_best_indices(mat, topk=topk, mask=mat_mask)
    return best_index



if __name__ == '__main__':    
    df = pd.read_csv("final.csv")
    nlp = spacy.load("en_core_web_lg") 
    df['spacy_sentence'] = df['sentence'].apply(lambda x: nlp(x)) 



