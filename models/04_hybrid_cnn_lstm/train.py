# models/04_hybrid_cnn_lstm/train.py
import sys
import os
import tensorflow as tf
import numpy as np
import json
from datetime import datetime

# Memory optimization settings
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Add parent directories to path
sys.path.append('../../')
sys.path.append('../')

try:
    from utils.data_loader import DataLoader
    from utils.common import setup_gpu
    from utils.metrics import ModelEvaluator
    from model import create_sequential_hybrid_model, create_simplified_parallel_model, create_ultra_light_model, compile_model
    from config import MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG, MODEL_SELECTION
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class MemoryOptimizedTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.data_loader = None
        self.model_name = "Hybrid_CNN_LSTM_Model"
        
    def setup_environment(self):
        """Setup memory-optimized training environment"""
        print("🚀 Setting up Memory-Optimized Training Environment...")
        
        # Clear any existing TensorFlow graphs
        tf.keras.backend.clear_session()
        
        # Setup GPU with memory growth
        setup_gpu()
        
        # Create directories
        os.makedirs(PATH_CONFIG.get('log_dir', 'training_logs'), exist_ok=True)
        os.makedirs(PATH_CONFIG.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
        os.makedirs(PATH_CONFIG.get('results_path', '../../results/model_performance'), exist_ok=True)
        
        print("✅ Environment setup complete!")
        
    def load_and_prepare_data(self):
        """Load data with memory optimization"""
        print("📊 Loading and preparing data...")
        
        data_path = PATH_CONFIG.get('data_path', '../../data/processed_data/')
        self.data_loader = DataLoader(data_path=data_path)
        self.data_loader.load_data()
        
        data_shapes = self.data_loader.get_data_shapes()
        print(f"Data shapes: {data_shapes}")
        
        # Convert to float32 to save memory (if using float64)
        self.data_loader.X_train = self.data_loader.X_train.astype(np.float32)
        self.data_loader.X_val = self.data_loader.X_val.astype(np.float32)
        self.data_loader.X_test = self.data_loader.X_test.astype(np.float32)
        
        print("✅ Data converted to float32 for memory efficiency")
        
        # Compute class weights
        if TRAINING_CONFIG.get('use_class_weights', True):
            class_weights = self.data_loader.get_class_weights()
            print(f"Class weights: {class_weights}")
        else:
            class_weights = None
            
        return data_shapes, class_weights
    
    def create_memory_optimized_model(self, input_shape):
        """Create memory-optimized model based on configuration"""
        print("🧠 Creating Memory-Optimized Model...")
        
        model_type = MODEL_SELECTION.get('model_type', 'sequential')
        
        if model_type == 'sequential':
            print("Using Sequential Hybrid Architecture (Memory Efficient)")
            self.model = create_sequential_hybrid_model(
                input_shape=input_shape,
                num_classes=2,
                config=MODEL_CONFIG
            )
        elif model_type == 'simplified_parallel':
            print("Using Simplified Parallel Architecture")
            self.model = create_simplified_parallel_model(
                input_shape=input_shape,
                num_classes=2,
                config=MODEL_CONFIG
            )
        else:
            print("Using Ultra-Light Architecture (Maximum Memory Efficiency)")
            self.model = create_ultra_light_model(
                input_shape=input_shape,
                num_classes=2
            )
        
        # Compile model
        learning_rate = TRAINING_CONFIG.get('learning_rate', 0.001)
        metrics = TRAINING_CONFIG.get('evaluation_config', {}).get('metrics', ['accuracy'])
        
        self.model = compile_model(
            self.model,
            learning_rate=learning_rate,
            metrics=metrics
        )
        
        return self.model
    
    def create_memory_safe_callbacks(self):
        """Create callbacks that don't use much memory"""
        print("⚙️ Creating memory-safe callbacks...")
        
        callbacks = []
        
        # Early Stopping
        if TRAINING_CONFIG.get('early_stopping', True):
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=TRAINING_CONFIG.get('early_stopping_monitor', 'val_loss'),
                patience=TRAINING_CONFIG.get('early_stopping_patience', 15),
                restore_best_weights=TRAINING_CONFIG.get('restore_best_weights', True),
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Learning Rate Scheduler
        if TRAINING_CONFIG.get('reduce_lr', True):
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=TRAINING_CONFIG.get('early_stopping_monitor', 'val_loss'),
                factor=TRAINING_CONFIG.get('reduce_lr_factor', 0.5),
                patience=TRAINING_CONFIG.get('reduce_lr_patience', 8),
                min_lr=TRAINING_CONFIG.get('reduce_lr_min', 1e-7),
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # Model Checkpoint - FIXED: Use .weights.h5 extension for weights-only saving
        base_path = PATH_CONFIG.get('model_save_path', 'best_hybrid_model.h5')
        # Remove old extensions and add the correct one
        base_name = os.path.splitext(base_path)[0]
        base_name = base_name.replace('.weights', '') # Remove if it exists
        model_save_path = base_name + '.weights.h5'
        print(f"Model weights will be saved to: {model_save_path}")

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor=TRAINING_CONFIG.get('early_stopping_monitor', 'val_loss'),
            save_best_only=True,
            save_weights_only=True,  # Only save weights to save memory
            verbose=1
        )
        callbacks.append(checkpoint)
        
        print(f"✅ Created {len(callbacks)} memory-safe callbacks")
        return callbacks
    
    def train_with_memory_management(self, class_weights=None):
        """Train model with memory management"""
        print("🎯 Starting Memory-Optimized Training...")
        
        callbacks = self.create_memory_safe_callbacks()
        
        # Get training parameters
        batch_size = TRAINING_CONFIG.get('batch_size', 16)
        epochs = TRAINING_CONFIG.get('epochs', 100)
        
        try:
            # Clear memory before training
            tf.keras.backend.clear_session()
            
            # Train with smaller batch size and memory optimization
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
            
            print("✅ Model training completed successfully!")
            return self.history
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Try reducing batch_size further or using the ultra-light model")
            raise
    
    def evaluate_model(self):
        """Evaluate model with memory considerations"""
        print("📈 Evaluating Model Performance...")
        
        # Load best weights if available - FIXED: Use correct file extension
        base_path = PATH_CONFIG.get('model_save_path', 'best_hybrid_model.h5')
        base_name = os.path.splitext(base_path)[0]
        base_name = base_name.replace('.weights', '')
        model_save_path = base_name + '.weights.h5'
        
        if os.path.exists(model_save_path):
            print("📥 Loading best saved weights...")
            self.model.load_weights(model_save_path)
        else:
            print(f"⚠️ Could not find saved weights at {model_save_path}. Evaluating with current model weights.")
        
        # Rest of the evaluate_model method remains the same...
        print("🔍 Making predictions (memory-safe)...")
        y_pred_proba = []
        batch_size = 8  # Small batch for evaluation
        
        for i in range(0, len(self.data_loader.X_test), batch_size):
            batch_x = self.data_loader.X_test[i:i+batch_size]
            batch_pred = self.model.predict(batch_x, verbose=0)
            y_pred_proba.append(batch_pred)
        
        y_pred_proba = np.vstack(y_pred_proba)
        y_pred = y_pred_proba.argmax(axis=1)
        
        # Evaluate
        evaluator = ModelEvaluator(self.model_name)
        if self.history:
            evaluator.set_history(self.history)
        evaluator.set_predictions(
            self.data_loader.y_test, 
            y_pred, 
            y_pred_proba[:, 1]
        )
        
        # Generate reports
        report, cm = evaluator.generate_classification_report()
        
        # Save results
        results_dir = PATH_CONFIG.get('results_path', '../../results/model_performance')
        os.makedirs(results_dir, exist_ok=True)
        
        # Plot results with lower DPI to save memory
        evaluator.plot_training_history(f'{results_dir}/{self.model_name}_training.png')
        evaluator.plot_confusion_matrix(cm, f'{results_dir}/{self.model_name}_cm.png')
        auc_score = evaluator.plot_roc_curve(f'{results_dir}/{self.model_name}_roc.png')
        
        evaluator.save_results(report, cm, results_dir)
        
        # Print results
        self._print_results(report, auc_score)
        
        return report, cm
    
    def _print_results(self, report, auc_score):
        """Print formatted results"""
        print("\n" + "="*60)
        print("🎉 MEMORY-OPTIMIZED HYBRID CNN-LSTM - RESULTS")
        print("="*60)
        print(f"📊 Accuracy:    {report['accuracy']:.3f}")
        print(f"🎯 Precision:   {report['1']['precision']:.3f}")
        print(f"🔍 Recall:      {report['1']['recall']:.3f}")
        print(f"⚖️  F1-Score:    {report['1']['f1-score']:.3f}")
        print(f"🛡️  Specificity: {report['specificity']:.3f}")
        if auc_score:
            print(f"📈 ROC AUC:     {auc_score:.3f}")
        print("="*60)

def main():
    """Main training function with memory optimization"""
    print("🚀 MEMORY-OPTIMIZED HYBRID CNN-LSTM TRAINING")
    print("="*50)
    
    # Initialize trainer
    trainer = MemoryOptimizedTrainer()
    
    try:
        # Setup environment
        trainer.setup_environment()
        
        # Load data
        data_shapes, class_weights = trainer.load_and_prepare_data()
        
        # Create model
        input_shape = (data_shapes['timesteps'], data_shapes['features'])
        trainer.create_memory_optimized_model(input_shape)
        
        # Display model architecture
        trainer.model.summary()
        
        # Train model
        trainer.train_with_memory_management(class_weights)
        
        # Evaluate model
        report, cm = trainer.evaluate_model()
        
        print("\n🎊 TRAINING COMPLETED SUCCESSFULLY!")
        print("📁 Results saved in:", PATH_CONFIG.get('results_path', '../../results/model_performance'))
        
        return trainer.model, trainer.history, report
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("1. Try reducing batch_size to 8 or 4")
        print("2. Use the ultra-light model architecture")
        print("3. Close other applications to free up memory")
        print("4. Restart your Python kernel and try again")
        return None, None, None

if __name__ == "__main__":
    model, history, report = main()