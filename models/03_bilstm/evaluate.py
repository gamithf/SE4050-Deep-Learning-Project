import sys
import os
sys.path.append('../../')

import numpy as np
import tensorflow as tf
from utils.data_loader import DataLoader
from utils.metrics import ModelEvaluator
from model import create_model, compile_model
import json

def load_trained_model(model_name, model_path):
    """Load a trained model"""
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        print(f"No trained model found at {model_path}")
        return None

def evaluate_model(model_name="Model"):
    """Evaluate trained model on test set"""
    
    # Setup
    print(f"🔍 Evaluating {model_name}...")
    
    # Load data
    data_loader = DataLoader()
    data_loader.load_data()
    data_shapes = data_loader.get_data_shapes()
    
    # Try to load trained model
    model_path = f'best_model.h5'
    model = load_trained_model(model_name, model_path)
    
    if model is None:
        print("❌ No trained model found. Please train the model first.")
        print("💡 Run: python train.py")
        return None
    
    # Make predictions
    print("📊 Making predictions...")
    y_pred_proba = model.predict(data_loader.X_test)
    y_pred = y_pred_proba.argmax(axis=1)
    
    # Evaluate
    evaluator = ModelEvaluator(model_name)
    evaluator.set_predictions(data_loader.y_test, y_pred, y_pred_proba[:, 1])
    
    # Generate reports and plots
    report, cm = evaluator.generate_classification_report()
    
    # Create results directory
    os.makedirs('../../results/model_performance/', exist_ok=True)
    
    # Plot results
    evaluator.plot_confusion_matrix(cm, f'../../results/model_performance/{model_name}_cm.png')
    evaluator.plot_roc_curve(f'../../results/model_performance/{model_name}_roc.png')
    evaluator.save_results(report, cm, '../../results/model_performance/')
    
    # Print results
    print(f"\n🎯 {model_name} Evaluation Results:")
    print("=" * 50)
    print(f"Accuracy:    {report['accuracy']:.3f}")
    print(f"Precision:   {report['1']['precision']:.3f}")
    print(f"Recall:      {report['1']['recall']:.3f}")
    print(f"F1-Score:    {report['1']['f1-score']:.3f}")
    print(f"Sensitivity: {report['sensitivity']:.3f}")
    print(f"Specificity: {report['specificity']:.3f}")
    if 'roc_auc' in report:
        print(f"ROC AUC:     {report['roc_auc']:.3f}")
    print("=" * 50)
    
    # Save detailed predictions
    predictions_data = {
        'y_true': data_loader.y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba[:, 1].tolist() if y_pred_proba.shape[1] > 1 else y_pred_proba.tolist()
    }
    
    with open(f'../../results/model_performance/{model_name}_predictions.json', 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    print(f"✅ Evaluation complete! Results saved to results/model_performance/")
    
    return report, model

def analyze_misclassifications(model_name, y_true, y_pred, X_test):
    """Analyze misclassified samples"""
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) > 0:
        print(f"\n🔍 Misclassification Analysis:")
        print(f"Total misclassified: {len(misclassified_idx)}/{len(y_true)} ({len(misclassified_idx)/len(y_true)*100:.1f}%)")
        
        # Analyze by class
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        print(f"False Positives (Control predicted as Condition): {false_positives}")
        print(f"False Negatives (Condition predicted as Control): {false_negatives}")
        
        return false_positives, false_negatives
    return 0, 0

if __name__ == "__main__":
    # Set your model name here
    MODEL_NAME = "CNN_Model"  # Change to "LSTM_Model", "BiLSTM_Model", "Hybrid_CNN_LSTM_Model"
    
    report, model = evaluate_model(MODEL_NAME)
    
    if report is not None:
        # Additional analysis
        data_loader = DataLoader()
        data_loader.load_data()
        
        y_pred_proba = model.predict(data_loader.X_test)
        y_pred = y_pred_proba.argmax(axis=1)
        
        analyze_misclassifications(MODEL_NAME, data_loader.y_test, y_pred, data_loader.X_test)