import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compare_all_models():
    """Compare performance of all four models"""
    model_names = ['CNN_Model', 'LSTM_Model', 'BiLSTM_Model', 'Hybrid_CNN_LSTM_Model']
    results = []
    
    for model_name in model_names:
        try:
            with open(f'../model_performance/{model_name}_report.json', 'r') as f:
                report = json.load(f)
                
            results.append({
                'Model': model_name,
                'Accuracy': report['accuracy'],
                'Precision': report['1']['precision'],
                'Recall': report['1']['recall'],
                'F1-Score': report['1']['f1-score'],
                'Sensitivity': report['sensitivity'],
                'Specificity': report['specificity'],
                'ROC-AUC': report.get('roc_auc', 'N/A')
            })
        except FileNotFoundError:
            print(f"Results for {model_name} not found")
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    print("\n=== MODEL COMPARISON ===")
    print(df.round(3))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy comparison
    axes[0,0].bar(df['Model'], df['Accuracy'], color='skyblue')
    axes[0,0].set_title('Model Accuracy Comparison')
    axes[0,0].set_ylabel('Accuracy')
    
    # F1-Score comparison
    axes[0,1].bar(df['Model'], df['F1-Score'], color='lightcoral')
    axes[0,1].set_title('Model F1-Score Comparison')
    axes[0,1].set_ylabel('F1-Score')
    
    # Sensitivity/Specificity comparison
    x = range(len(df))
    width = 0.35
    axes[1,0].bar([i - width/2 for i in x], df['Sensitivity'], width, label='Sensitivity', color='lightgreen')
    axes[1,0].bar([i + width/2 for i in x], df['Specificity'], width, label='Specificity', color='orange')
    axes[1,0].set_title('Sensitivity vs Specificity')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(df['Model'])
    axes[1,0].legend()
    
    # ROC-AUC comparison
    if 'ROC-AUC' in df.columns and not all(df['ROC-AUC'] == 'N/A'):
        auc_scores = [score if score != 'N/A' else 0 for score in df['ROC-AUC']]
        axes[1,1].bar(df['Model'], auc_scores, color='purple')
        axes[1,1].set_title('ROC-AUC Comparison')
        axes[1,1].set_ylabel('AUC Score')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

if __name__ == "__main__":
    comparison_df = compare_all_models()
    comparison_df.to_csv('model_comparison_results.csv', index=False)