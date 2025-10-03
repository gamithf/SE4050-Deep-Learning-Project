# models/04_hybrid_cnn_lstm/train.py
import sys
import os
import tensorflow as tf
import numpy as np
import json
from datetime import datetime

# Add parent directories to path
sys.path.append('../../')
sys.path.append('../')

# Import with error handling
try:
    from utils.data_loader import DataLoader
    from utils.common import setup_gpu
    from utils.metrics import ModelEvaluator
    from model import create_hybrid_model, create_sequential_hybrid_model, compile_model
    from config import MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG, MODEL_SELECTION
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you have the required files in the correct locations")
    sys.exit(1)

class HybridModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.data_loader = None
        self.model_name = "Hybrid_CNN_LSTM_Model"
        
    def setup_environment(self):
        """Setup training environment"""
        print("🚀 Setting up Hybrid CNN-LSTM Training Environment...")
        
        # Setup GPU
        setup_gpu()
        
        # Create directories with error handling
        try:
            os.makedirs(PATH_CONFIG.get('log_dir', 'training_logs'), exist_ok=True)
            os.makedirs(PATH_CONFIG.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
            os.makedirs(PATH_CONFIG.get('results_path', '../../results/model_performance'), exist_ok=True)
            print("✅ Environment setup complete!")
        except Exception as e:
            print(f"❌ Directory creation failed: {e}")
            raise
        
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        print("📊 Loading and preparing data...")
        
        data_path = PATH_CONFIG.get('data_path', '../../data/processed_data/')
        self.data_loader = DataLoader(data_path=data_path)
        self.data_loader.load_data()
        
        data_shapes = self.data_loader.get_data_shapes()
        print(f"Data shapes: {data_shapes}")
        
        # Compute class weights for imbalance
        if TRAINING_CONFIG.get('use_class_weights', True):
            class_weights = self.data_loader.get_class_weights()
            print(f"Class weights: {class_weights}")
        else:
            class_weights = None
            
        return data_shapes, class_weights
    
    def create_model(self, input_shape):
        """Create the hybrid model based on configuration"""
        print("🧠 Creating Hybrid CNN-LSTM Model...")
        
        model_type = MODEL_SELECTION.get('model_type', 'advanced_parallel')
        
        if model_type == 'advanced_parallel':
            print("Using Advanced Parallel Hybrid Architecture")
            self.model = create_hybrid_model(
                input_shape=input_shape,
                num_classes=2,
                config=MODEL_CONFIG
            )
        else:
            print("Using Sequential Hybrid Architecture")
            self.model = create_sequential_hybrid_model(
                input_shape=input_shape,
                num_classes=2,
                config=MODEL_CONFIG
            )
        
        # Get metrics safely
        eval_config = TRAINING_CONFIG.get('evaluation_config', {})
        metrics = eval_config.get('metrics', ['accuracy'])
        
        # Compile model
        learning_rate = TRAINING_CONFIG.get('learning_rate', 0.001)
        self.model = compile_model(
            self.model,
            learning_rate=learning_rate,
            metrics=metrics
        )
        
        return self.model
    
    def create_callbacks(self):
        """Create training callbacks"""
        print("⚙️ Creating training callbacks...")
        
        callbacks = []
        
        # Early Stopping
        if TRAINING_CONFIG.get('early_stopping', True):
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=TRAINING_CONFIG.get('early_stopping_monitor', 'val_loss'),
                patience=TRAINING_CONFIG.get('early_stopping_patience', 20),
                restore_best_weights=TRAINING_CONFIG.get('restore_best_weights', True),
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Learning Rate Scheduler
        if TRAINING_CONFIG.get('reduce_lr', True):
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=TRAINING_CONFIG.get('early_stopping_monitor', 'val_loss'),
                factor=TRAINING_CONFIG.get('reduce_lr_factor', 0.5),
                patience=TRAINING_CONFIG.get('reduce_lr_patience', 10),
                min_lr=TRAINING_CONFIG.get('reduce_lr_min', 1e-7),
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # Model Checkpoint
        model_save_path = PATH_CONFIG.get('model_save_path', 'best_hybrid_model.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor=TRAINING_CONFIG.get('early_stopping_monitor', 'val_loss'),
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # CSV Logger
        log_dir = PATH_CONFIG.get('log_dir', 'training_logs')
        csv_logger = tf.keras.callbacks.CSVLogger(
            f"{log_dir}/training_log.csv"
        )
        callbacks.append(csv_logger)
        
        print(f"✅ Created {len(callbacks)} callbacks")
        return callbacks
    
    def train_model(self, class_weights=None):
        """Train the hybrid model"""
        print("🎯 Starting Model Training...")
        
        callbacks = self.create_callbacks()
        
        # Get training parameters with defaults
        batch_size = TRAINING_CONFIG.get('batch_size', 32)
        epochs = TRAINING_CONFIG.get('epochs', 100)
        
        # Start training
        self.history = self.model.fit(
            self.data_loader.X_train,
            self.data_loader.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                self.data_loader.X_val, 
                self.data_loader.y_val
            ),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        print("✅ Model training completed!")
        return self.history
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("📈 Evaluating Model Performance...")
        
        # Load best model if available
        model_save_path = PATH_CONFIG.get('model_save_path', 'best_hybrid_model.h5')
        if os.path.exists(model_save_path):
            print("📥 Loading best saved model...")
            self.model = tf.keras.models.load_model(model_save_path)
        
        # Make predictions
        y_pred_proba = self.model.predict(self.data_loader.X_test)
        y_pred = y_pred_proba.argmax(axis=1)
        
        # Evaluate
        evaluator = ModelEvaluator(self.model_name)
        if self.history:
            evaluator.set_history(self.history)
        evaluator.set_predictions(
            self.data_loader.y_test, 
            y_pred, 
            y_pred_proba[:, 1]  # Probability for class 1
        )
        
        # Generate reports and plots
        report, cm = evaluator.generate_classification_report()
        
        # Save detailed results
        results_dir = PATH_CONFIG.get('results_path', '../../results/model_performance')
        os.makedirs(results_dir, exist_ok=True)
        
        # Plot results
        evaluator.plot_training_history(f'{results_dir}/{self.model_name}_training.png')
        evaluator.plot_confusion_matrix(cm, f'{results_dir}/{self.model_name}_cm.png')
        auc_score = evaluator.plot_roc_curve(f'{results_dir}/{self.model_name}_roc.png')
        
        # Save results
        evaluator.save_results(report, cm, results_dir)
        
        # Save model architecture
        if TRAINING_CONFIG.get('save_model_architecture', True):
            with open(f'{results_dir}/{self.model_name}_architecture.txt', 'w') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Print comprehensive results
        self._print_results(report, auc_score)
        
        return report, cm
    
    def _print_results(self, report, auc_score):
        """Print formatted results"""
        print("\n" + "="*70)
        print("🎉 HYBRID CNN-LSTM MODEL - FINAL RESULTS")
        print("="*70)
        print(f"📊 Overall Accuracy:    {report['accuracy']:.3f}")
        print(f"🎯 Precision:           {report['1']['precision']:.3f}")
        print(f"🔍 Recall (Sensitivity): {report['1']['recall']:.3f}")
        print(f"⚖️  F1-Score:            {report['1']['f1-score']:.3f}")
        print(f"🛡️  Specificity:         {report['specificity']:.3f}")
        if auc_score:
            print(f"📈 ROC AUC:             {auc_score:.3f}")
        print("="*70)
        
        # Additional metrics
        print(f"📋 Classification Report:")
        print(f"   - Control (0): Precision {report['0']['precision']:.3f}, Recall {report['0']['recall']:.3f}")
        print(f"   - Condition (1): Precision {report['1']['precision']:.3f}, Recall {report['1']['recall']:.3f}")
        print("="*70)
    
    def save_training_artifacts(self):
        """Save training artifacts for reproducibility"""
        print("💾 Saving training artifacts...")
        
        artifacts = {
            'training_config': TRAINING_CONFIG,
            'model_config': MODEL_CONFIG,
            'path_config': PATH_CONFIG,
            'training_timestamp': datetime.now().isoformat(),
            'environment_info': self._get_environment_info()
        }
        
        results_path = PATH_CONFIG.get('results_path', '../../results/model_performance')
        with open(f"{results_path}/training_artifacts.json", 'w') as f:
            json.dump(artifacts, f, indent=2)
        
        print("✅ Training artifacts saved!")
    
    def _get_environment_info(self):
        """Get environment information"""
        import platform
        return {
            'python_version': platform.python_version(),
            'tensorflow_version': tf.__version__,
            'platform': platform.platform(),
            'processor': platform.processor()
        }

def main():
    """Main training function"""
    print("🚀 HYBRID CNN-LSTM DEPRESSION DETECTION MODEL")
    print("="*60)
    
    # Initialize trainer
    trainer = HybridModelTrainer()
    
    try:
        # Setup environment
        trainer.setup_environment()
        
        # Load data
        data_shapes, class_weights = trainer.load_and_prepare_data()
        
        # Create model
        input_shape = (data_shapes['timesteps'], data_shapes['features'])
        trainer.create_model(input_shape)
        
        # Display model architecture
        trainer.model.summary()
        
        # Train model
        trainer.train_model(class_weights)
        
        # Evaluate model
        report, cm = trainer.evaluate_model()
        
        # Save artifacts
        trainer.save_training_artifacts()
        
        print("\n🎊 HYBRID CNN-LSTM TRAINING COMPLETED SUCCESSFULLY!")
        results_path = PATH_CONFIG.get('results_path', '../../results/model_performance')
        print("📁 Results saved in:", results_path)
        
        return trainer.model, trainer.history, report
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, history, report = main()