import sys
import os
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

sys.path.append('../../')
sys.path.append('../')

from utils.data_loader import DataLoader
from utils.metrics import ModelEvaluator
from config import PATH_CONFIG, MODEL_CONFIG
from model import create_sequential_hybrid_model

class HybridModelEvaluator:
    def __init__(self):
        self.model = None
        self.data_loader = None
        self.model_name = "Hybrid_CNN_LSTM_Model"
        
    def load_model_and_data(self):
        """Load data, re-create model architecture, and then load the trained weights."""
        print("📥 Loading data and model...")
        
        # 1. Load data to determine the model's input shape
        self.data_loader = DataLoader(data_path=PATH_CONFIG['data_path'])
        self.data_loader.load_data()
        data_shapes = self.data_loader.get_data_shapes()
        input_shape = (data_shapes['timesteps'], data_shapes['features'])
        
        # 2. Re-create the exact same model architecture used for training
        print("🧠 Re-creating model architecture...")
        self.model = create_sequential_hybrid_model(
            input_shape=input_shape,
            num_classes=data_shapes.get('num_classes', 2),
            config=MODEL_CONFIG
        )
        
        # FIXED: Explicitly build the model to define its input/output shapes for analysis
        self.model.build(input_shape=(None, *input_shape))
        print("✅ Model architecture created and built.")

        # 3. Load the saved weights into the new model structure
        model_path = PATH_CONFIG['model_save_path']
        if os.path.exists(model_path):
            print(f"⚖️ Loading trained weights from {model_path}...")
            self.model.load_weights(model_path)
            print("✅ Weights loaded successfully.")
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}. Please train the model first.")
            
        return self.model
    
    # ... (comprehensive_evaluation, _feature_importance_analysis, etc., remain the same) ...
    def comprehensive_evaluation(self):
        """Perform comprehensive model evaluation"""
        print("🔍 Performing comprehensive evaluation...")
        
        y_pred_proba = self.model.predict(self.data_loader.X_test)
        y_pred = y_pred_proba.argmax(axis=1)
        y_true = self.data_loader.y_test
        
        evaluator = ModelEvaluator(self.model_name)
        evaluator.set_predictions(y_true, y_pred, y_pred_proba[:, 1])
        
        report, cm = evaluator.generate_classification_report()
        
        self._feature_importance_analysis()
        self._temporal_pattern_analysis()
        self._confidence_analysis(y_pred_proba, y_true)
        
        return report, cm, y_pred_proba, y_pred, y_true
    
    def _feature_importance_analysis(self):
        """Analyze feature importance using gradient-based methods"""
        print("📊 Analyzing feature importance...")
        try:
            layer_outputs = [layer.output for layer in self.model.layers[:8]]
            activation_model = tf.keras.models.Model(
                inputs=self.model.input, 
                outputs=layer_outputs
            )
            
            sample_input = self.data_loader.X_test[:1]
            activations = activation_model.predict(sample_input)
            
            print(f"✅ Feature analysis completed for {len(activations)} layers")
            
        except Exception as e:
            print(f"⚠️  Feature importance analysis skipped: {e}")
    
    def _temporal_pattern_analysis(self):
        """Analyze temporal patterns in predictions"""
        print("⏰ Analyzing temporal patterns...")
        try:
            all_predictions = self.model.predict(self.data_loader.X_test)
            confidence_scores = np.max(all_predictions, axis=1)
            avg_confidence = np.mean(confidence_scores)
            
            print(f"📈 Average prediction confidence: {avg_confidence:.3f}")
            print(f"🔍 Confidence range: {np.min(confidence_scores):.3f} - {np.max(confidence_scores):.3f}")
            
        except Exception as e:
            print(f"⚠️  Temporal pattern analysis skipped: {e}")
    
    def _confidence_analysis(self, y_pred_proba, y_true):
        """Analyze prediction confidence"""
        print("🎯 Analyzing prediction confidence...")
        
        confidence = np.max(y_pred_proba, axis=1)
        correct_predictions = (y_pred_proba.argmax(axis=1) == y_true)
        
        correct_confidence = confidence[correct_predictions]
        incorrect_confidence = confidence[~correct_predictions]
        
        print(f"✅ Correct predictions: {len(correct_confidence)}")
        print(f"❌ Incorrect predictions: {len(incorrect_confidence)}")
        
        if len(correct_confidence) > 0:
            print(f"📊 Avg confidence (correct): {np.mean(correct_confidence):.3f}")
        if len(incorrect_confidence) > 0:
            print(f"📊 Avg confidence (incorrect): {np.mean(incorrect_confidence):.3f}")

    def generate_detailed_report(self, report, cm, y_pred_proba, y_pred, y_true):
        """Generate detailed evaluation report"""
        print("📋 Generating detailed evaluation report...")
        
        detailed_report = {
            'model_name': self.model_name,
            'timestamp': np.datetime64('now').astype(str),
            'overall_metrics': {
                'accuracy': report['accuracy'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1_score': report['1']['f1-score'],
                'specificity': report['specificity'],
                'auc_roc': report.get('roc_auc', 'N/A')
            },
            'class_wise_metrics': {
                'control': report['0'],
                'condition': report['1']
            },
            'confusion_matrix': cm.tolist(),
            'dataset_info': {
                'test_samples': len(y_true),
                'class_distribution': {
                    # FIXED: Cast NumPy int64 to standard Python int
                    'control': int(np.sum(y_true == 0)),
                    'condition': int(np.sum(y_true == 1))
                }
            },
            'model_architecture': {
                'total_layers': len(self.model.layers),
                # FIXED: Cast NumPy int64 to standard Python int
                'trainable_params': int(self.model.count_params()),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape
            }
        }
        
        report_path = f"{PATH_CONFIG['results_path']}/{self.model_name}_detailed_report.json"
        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"✅ Detailed report saved to: {report_path}")
        return detailed_report

    # ... (plot_advanced_metrics and main function remain the same) ...
    def plot_advanced_metrics(self, y_true, y_pred_proba):
        """Plot advanced evaluation metrics"""
        print("📈 Generating advanced metric plots...")
        
        plots_dir = f"{PATH_CONFIG['results_path']}/advanced_plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        # We need the full probability array for this plot, not just class 1
        full_pred_proba = np.hstack([(1-y_pred_proba).reshape(-1,1), y_pred_proba.reshape(-1,1)])
        plt.hist(full_pred_proba[y_true == 0, 1], alpha=0.7, label='Control (Prob for Condition)', bins=20)
        plt.hist(full_pred_proba[y_true == 1, 1], alpha=0.7, label='Condition (Prob for Condition)', bins=20)
        plt.xlabel('Predicted Probability for Condition')
        plt.ylabel('Frequency')
        plt.title('Probability Distribution by True Class')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        plt.plot(prob_pred, prob_true, 's-', label='Hybrid CNN-LSTM')
        plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{self.model_name}_advanced_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Advanced plots saved to: {plots_dir}")

def main():
    """Main evaluation function"""
    print("🔍 HYBRID CNN-LSTM MODEL EVALUATION")
    print("="*50)
    
    evaluator = HybridModelEvaluator()
    
    try:
        evaluator.load_model_and_data()
        
        report, cm, y_pred_proba, y_pred, y_true = evaluator.comprehensive_evaluation()
        
        detailed_report = evaluator.generate_detailed_report(report, cm, y_pred_proba, y_pred, y_true)
        
        evaluator.plot_advanced_metrics(y_true, y_pred_proba[:, 1])
        
        print("\n" + "="*60)
        print("🎯 EVALUATION SUMMARY")
        print("="*60)
        print(f"📊 Accuracy: {report['accuracy']:.3f}")
        print(f"🎯 F1-Score: {report['1']['f1-score']:.3f}")
        print(f"🔍 Sensitivity: {report['1']['recall']:.3f}")
        print(f"🛡️  Specificity: {report['specificity']:.3f}")
        auc_score = report.get('roc_auc') 
        if auc_score:
            print(f"📈 AUC-ROC: {auc_score:.3f}")
        print("="*60)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved in: {PATH_CONFIG['results_path']}")
        
        return detailed_report
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    report = main()