import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns

df = pd.read_csv("Final_implementation\classification_evaluation_dataset.csv", sep='|')

#DeepSeekOnto
prediction_cols_deepseek_onto = ["DeepSeek_onto_CoT_result_noFormat","DeepSeek_onto_Few-shot_result_noFormat",
                   "DeepSeek_onto_CoT_result_twoCharacteristic","DeepSeek_onto_Few-shot_result_twoCharacteristic",
                   "DeepSeek_onto_CoT_result_allCharacteristics", "DeepSeek_onto_Few-shot_result_allCharacteristics"]
#GeminiOnto
prediction_cols_gemini_onto = ["Gemini_onto_CoT_result_noFormat","Gemini_onto_Few-shot_result_noFormat",
                   "Gemini_onto_CoT_result_twoCharacteristic","Gemini_onto_Few-shot_result_twoCharacteristic",
                   "Gemini_onto_CoT_result_allCharacteristics","Gemini_onto_Few-shot_result_allCharacteristics"]
#Best Result comparison
prediction_cols_comparison = ["DeepSeek_onto_CoT_result_noFormat","DeepSeek_onto_Few-shot_result_noFormat",
                   "Gemini_onto_CoT_result_twoCharacteristic","Gemini_onto_Few-shot_result_twoCharacteristic",
                   "DeepSeek_CoT_result","DeepSeek_Few-shot_result",
                   "Gemini_CoT_result","Gemini_Few-shot_result"]

label_col = "toxicity"


reports = {}
missing_info = {}

def visualization(prediction_cols):
    for col in prediction_cols:
        df_pair = df[[label_col, col]]

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


    metrics_df = pd.DataFrame(reports).T  #Models as row

    plt.figure(figsize=(10, len(metrics_df)*0.6))
    sns.heatmap(metrics_df, annot=True, cmap='YlOrBr', fmt=".2f", cbar=True)
    plt.title("Classification Report per Esperimento")
    plt.xlabel("Metriche")
    plt.ylabel("Esperimenti")
    plt.tight_layout()
    plt.show()

#visualization(prediction_cols_deepseek_onto)
#visualization(prediction_cols_gemini_onto)
visualization(prediction_cols_comparison)