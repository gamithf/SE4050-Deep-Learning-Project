# models/04_hybrid_cnn_lstm/config.py
"""
Advanced Configuration for Hybrid CNN-LSTM Model
Team Member: You can experiment with these hyperparameters
"""

# ===== MODEL ARCHITECTURE CONFIGURATION =====
MODEL_CONFIG = {
    # CNN Configuration
    'conv_filters': [64, 128, 256],      # Number of filters in each conv layer
    'kernel_sizes': [7, 5, 3],           # Kernel sizes for conv layers
    'use_multi_scale': True,             # Use multi-scale CNN approach
    
    # LSTM Configuration  
    'lstm_units': [128, 64],             # Units in each LSTM layer
    'use_bidirectional': False,          # Set to True for BiLSTM instead of LSTM
    
    # Dense Layers Configuration
    'dense_units': [512, 256, 128],      # Units in dense layers
    'activation': 'relu',                # Activation function
    
    # Regularization - ADD THE MISSING KEY
    'dropout_rates': [0.2, 0.2, 0.3, 0.3, 0.5, 0.5, 0.3],
    'l2_reg': 0.001,                     # ADD THIS LINE - L2 regularization strength
    'use_batch_norm': True,              # Use batch normalization
    
    # Feature Fusion
    'fusion_method': 'concatenate',      # 'concatenate' or 'add'
    'use_attention': False,              # Experimental: Use attention mechanism
}

# ===== TRAINING CONFIGURATION =====
TRAINING_CONFIG = {
    'batch_size': 32,                    # Training batch size
    'epochs': 150,                       # Maximum training epochs
    'learning_rate': 0.001,              # Initial learning rate
    'learning_rate_schedule': True,      # Use learning rate scheduling
    
    # Early Stopping
    'early_stopping': True,
    'early_stopping_patience': 20,
    'early_stopping_monitor': 'val_loss',
    'restore_best_weights': True,
    
    # Learning Rate Reduction
    'reduce_lr': True,
    'reduce_lr_patience': 10,
    'reduce_lr_factor': 0.5,
    'reduce_lr_min': 1e-7,
    
    # Class weights for imbalance
    'use_class_weights': True,
    
    # Gradient Clipping (for stability)
    'gradient_clip': True,
    'clip_value': 1.0,
    
    # Evaluation metrics - ADD THIS SECTION
    'evaluation_config': {
        'metrics': ['accuracy', 'precision', 'recall', 'auc']
    },
    
    # Model saving
    'save_model_architecture': True,
    'use_tensorboard': False,  # Set to False to avoid TensorBoard issues
}

# ===== MODEL SELECTION =====
MODEL_SELECTION = {
    'model_type': 'advanced_parallel',   # 'advanced_parallel' or 'sequential'
    'use_pretrained': False,
}

# ===== EVALUATION CONFIGURATION =====
EVALUATION_CONFIG = {
    'save_plots': True,
    'plot_dpi': 300,
    'generate_report': True,
    'save_predictions': True,
    'confidence_threshold': 0.5,
    
    # Metrics to track
    'metrics': ['accuracy', 'precision', 'recall', 'auc'],
    
    # Cross-validation (optional)
    'use_cross_validation': False,
    'cv_folds': 5,
}

# ===== PATHS AND FILES =====
PATH_CONFIG = {
    'data_path': '../../data/processed_data/',
    'results_path': '../../results/model_performance/',
    'model_save_path': 'best_hybrid_model.h5',
    'log_dir': 'training_logs/',
    'checkpoint_dir': 'checkpoints/',
}

# ===== ADVANCED FEATURES =====
ADVANCED_CONFIG = {
    # Data augmentation for time series
    'use_data_augmentation': False,
    'augmentation_methods': ['time_warp', 'noise_injection'],
    
    # Ensemble learning (optional)
    'use_ensemble': False,
    'ensemble_models': 3,
    
    # Transfer learning (if applicable)
    'use_transfer_learning': False,
    'pretrained_path': None,
    
    # Hyperparameter tuning
    'tune_hyperparameters': False,
    'tuning_method': 'bayesian',  # 'grid', 'random', 'bayesian'
}

# ===== EXPERIMENT TRACKING =====
EXPERIMENT_CONFIG = {
    'experiment_name': 'Hybrid_CNN_LSTM_Depression_Detection',
    'track_experiment': True,
    'use_tensorboard': False,  # Disabled to avoid issues
    'save_model_architecture': True,
    
    # Model versioning
    'version': '1.0.0',
    'description': 'Advanced Hybrid CNN-LSTM for depression classification',
}