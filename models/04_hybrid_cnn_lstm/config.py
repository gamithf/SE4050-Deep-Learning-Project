# models/04_hybrid_cnn_lstm/config.py
"""
Memory-Optimized Configuration for Hybrid CNN-LSTM Model
"""

# ===== MODEL ARCHITECTURE CONFIGURATION =====
MODEL_CONFIG = {
    # CNN Configuration - Reduced sizes for memory efficiency
    'conv_filters': [32, 64, 128],      # REDUCED: Was [64, 128, 256]
    'kernel_sizes': [5, 3, 3],          # Smaller kernels
    'use_multi_scale': False,           # Disabled to save memory
    
    # LSTM Configuration - Reduced units
    'lstm_units': [64, 32],             # REDUCED: Was [128, 64]
    
    # Dense Layers Configuration - Reduced units
    'dense_units': [128, 64, 32],       # REDUCED: Was [512, 256, 128]
    'activation': 'relu',
    
    # Regularization
    'dropout_rates': [0.2, 0.2, 0.3, 0.3, 0.4, 0.4],
    'l2_reg': 0.001,
    'use_batch_norm': True,
    
    # Feature Fusion
    'fusion_method': 'concatenate',
    'use_attention': False,
}

# ===== TRAINING CONFIGURATION =====
TRAINING_CONFIG = {
    'batch_size': 16,                   # REDUCED: Was 32 (HALVED for memory)
    'epochs': 100,                      # Reduced epochs
    'learning_rate': 0.001,
    'learning_rate_schedule': True,
    
    # Early Stopping
    'early_stopping': True,
    'early_stopping_patience': 15,
    'early_stopping_monitor': 'val_loss',
    'restore_best_weights': True,
    
    # Learning Rate Reduction
    'reduce_lr': True,
    'reduce_lr_patience': 8,
    'reduce_lr_factor': 0.5,
    'reduce_lr_min': 1e-7,
    
    # Class weights for imbalance
    'use_class_weights': True,
    
    # Gradient Clipping
    'gradient_clip': False,             # Disabled for stability
    
    # Evaluation metrics
    'evaluation_config': {
        'metrics': ['accuracy']         # Simplified to just accuracy
    },
    
    # Model saving
    'save_model_architecture': True,
    'use_tensorboard': False,
}

# ===== MODEL SELECTION =====
MODEL_SELECTION = {
    'model_type': 'sequential',         # CHANGED: Use sequential (less memory)
    'use_pretrained': False,
}

# ===== EVALUATION CONFIGURATION =====
EVALUATION_CONFIG = {
    'save_plots': True,
    'plot_dpi': 200,                    # Lower DPI to save memory
    'generate_report': True,
    'save_predictions': True,
    'confidence_threshold': 0.5,
    'metrics': ['accuracy'],
}

# ===== PATHS AND FILES =====
PATH_CONFIG = {
    'data_path': '../../data/processed_data/',
    'results_path': '../../results/model_performance/',
    'model_save_path': 'best_hybrid_model.weights.h5', 
    'log_dir': 'training_logs/',
    'checkpoint_dir': 'checkpoints/',
}