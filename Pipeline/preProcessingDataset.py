import pandas as pd
import os
import numpy as np

#PreProcessing del dataset di classificazione della tossicità
dataset1 = pd.read_csv("benchmark_classification.csv", sep='|')

# #Questo codice binarizza toxicity in 0 e 1
def binarize(value):
    if value > 0.5:
        return 1
    else: return 0

dataset1['toxicity']=dataset1['toxicity'].apply(binarize)

#Aggiungo le colonne utili per la valutazione
dataset1['Gemini_onto_Few-shot_prompt'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Gemini_onto_Few-shot_classification'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Gemini_onto_Few-shot_classification_float'] = np.nan
dataset1['Gemini_onto_CoT_prompt'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Gemini_onto_CoT_classification'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Gemini_onto_CoT_classification_float'] = np.nan
dataset1['Gemini_Plain_classification'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Gemini_techniques_suggested'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Gemini_Few-shot_prompt'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Gemini_Few-shot_classification'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Gemini_CoT_prompt'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Gemini_CoT_classification'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Deepseek_onto_Few-shot_prompt'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Deepseek_onto_Few-shot_classification'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Deepseek_onto_CoT_prompt'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Deepseek_onto_CoT_classification'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Deepseek_Plain_classification'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Deepseek_techniques_suggested'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Deepseek_Few-shot_prompt'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Deepseek_Few-shot_classification'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Deepseek_CoT_prompt'] = pd.Series([""] * len(dataset1), dtype=object)
dataset1['Deepseek_CoT_classification'] = pd.Series([""] * len(dataset1), dtype=object)

# #Prendo 100 elementi con una distribuzione 60/40
classe_0 = dataset1[dataset1['toxicity'] == 0]
classe_1 = dataset1[dataset1['toxicity'] == 1]

n = min(len(classe_0), len(classe_1), 50)
print(n)
sample_0 = classe_0.sample(n=100-n, random_state=42)
sample_1 = classe_1.sample(n=n, random_state=42)
dataset1_preprocessato = pd.concat([sample_0, sample_1])
#Mescola le righe per evitare blocchi di classi
dataset1_preprocessato = dataset1_preprocessato.sample(frac=1, random_state=42).reset_index(drop=True)
dataset1_preprocessato.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')
