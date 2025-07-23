import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns

df = pd.read_csv("Final_implementation\classification_evaluation_dataset.csv", sep='|')

prediction_cols = ["DeepSeek_onto_CoT_result_noFormat","DeepSeek_onto_Few-shot_result_noFormat","Gemini_onto_CoT_result_noFormat","Gemini_onto_Few-shot_result_noFormat","DeepSeek_onto_CoT_result_allCharacteristics", "DeepSeek_onto_Few-shot_result_allCharacteristics","Gemini_onto_Few-shot_result_allCharacteristics","Gemini_onto_Few-shot_result_allCharacteristics", "DeepSeek_onto_CoT_result_twoCharacteristic","DeepSeek_onto_Few-shot_result_twoCharacteristic","Gemini_onto_CoT_result_twoCharacteristic","Gemini_onto_Few-shot_result_twoCharacteristic","DeepSeek_Plain_classification","Gemini_Plain_classification"]
label_col = "toxicity"


reports = {}
missing_info = {}

for col in prediction_cols:
    df_pair = df[[label_col, col]]

    nan_rows = df_pair[df_pair.isna().any(axis=1)]
    #Stampa indici dei nan
    nan_indices = df_pair[df_pair.isna().any(axis=1)].index
    nan_indices_list = nan_indices.tolist()
    print(nan_indices)
    ###
    num_nans = len(nan_rows)
    num_label1_in_nans = nan_rows[label_col].eq(1).sum()

    # Salva info
    missing_info[col] = {
        'Righe NaN': num_nans,
        'Di cui con label=1': num_label1_in_nans
    }

    y_true_clean = df_pair[label_col]
    y_pred_clean = df_pair[col].apply(lambda x: x if pd.notna(x) else -1)

    report = classification_report(
        y_true_clean, y_pred_clean, output_dict=True, zero_division=0, labels=[0, 1]
    )

    metrics = report['macro avg']
    
    reports[col] = {
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1-score']
    }

# Converto in DataFrame
results_df = pd.DataFrame(reports).T 
missing_df = pd.DataFrame(missing_info).T

print("\nRiepilogo righe NaN:")
print(missing_df)

################################
#TABLE
################################
#VIsualizza il report
metrics_df = pd.DataFrame(reports).T  # Modelli come righe

plt.figure(figsize=(10, len(metrics_df)*0.6))
sns.heatmap(metrics_df, annot=True, cmap='YlOrBr', fmt=".2f", cbar=True)
plt.title("Classification Report per Esperimento")
plt.xlabel("Metriche")
plt.ylabel("Esperimenti")
plt.tight_layout()
#plt.show()

#########################
#BOXPLOT
#########################
metrics_long_df = metrics_df.reset_index().melt(
    id_vars='index', var_name='Metrica', value_name='Valore'
).rename(columns={'index':'Esperimento'})

# Identifica il tipo dell’esperimento da ogni nome di colonna
def get_tipo(exp_name):
    if 'Few-shot' in exp_name:
        return 'Few-shot'
    if 'CoT' in exp_name:
        return 'CoT'
    if 'Plain' in exp_name:
        return 'Plain'
    return 'Altro'

metrics_long_df['Tipo'] = metrics_long_df['Esperimento'].map(get_tipo)

# 3. Ordina metriche per mediana (opzionale)
order = metrics_long_df.groupby('Metrica')['Valore'].median().sort_values(ascending=False).index

# 4. Plot
plt.figure(figsize=(12, 7))
sns.boxplot(
    data=metrics_long_df,
    x='Metrica', y='Valore',
    palette='Set2',
    order=order
)
sns.stripplot(
    data=metrics_long_df,
    x='Metrica', y='Valore',
    hue='Tipo',
    dodge=True,
    jitter=True,
    alpha=0.7
)

# Miglioramenti estetici
plt.title("BoxPlot")
plt.ylabel("Metric value")
plt.xlabel("Metric name")
#plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Prompt technique', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()

# metrics_df = pd.DataFrame({
#     model: {
#         'precision': report['macro avg']['precision'],
#         'recall': report['macro avg']['recall'],
#         'f1-score': report['macro avg']['f1-score']
#     }
#     for model, report in reports.items()
# }).T

# # Plot a barre
# metrics_df.plot(kind='bar', figsize=(10, 6))
# plt.title('Macro Average Metrics per modello')
# plt.ylabel('Score')
# plt.ylim(0, 1.05)
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()