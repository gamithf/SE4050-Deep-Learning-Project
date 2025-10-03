import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

def create_callbacks(model_name, patience=15, monitor='val_loss'):
    """Create common callbacks for all models"""
    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'models/{model_name}/best_model.h5',
            monitor=monitor,
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def setup_gpu():
    """Setup GPU configuration for consistent training"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU setup complete: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU available, using CPU")