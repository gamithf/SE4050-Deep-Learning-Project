# models/04_hybrid_cnn_lstm/model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    BatchNormalization, Concatenate, GlobalMaxPooling1D, 
    GlobalAveragePooling1D, Flatten
)
from tensorflow.keras.regularizers import l2

def create_sequential_hybrid_model(input_shape, num_classes=2, config=None):
    """
    Memory-optimized Sequential Hybrid CNN-LSTM
    CNN -> LSTM -> Dense (uses less memory than parallel)
    """
    # Safe configuration with defaults
    if config is None:
        config = {}
    
    conv_filters = config.get('conv_filters', [32, 64, 128])
    kernel_sizes = config.get('kernel_sizes', [5, 3, 3])
    lstm_units = config.get('lstm_units', [64, 32])
    dense_units = config.get('dense_units', [128, 64, 32])
    dropout_rates = config.get('dropout_rates', [0.2, 0.2, 0.3, 0.3, 0.4, 0.4])
    l2_reg = config.get('l2_reg', 0.001)
    
    print("🧠 Creating Memory-Optimized Sequential Hybrid CNN-LSTM...")
    
    model = Sequential([
        # ===== CNN FEATURE EXTRACTION =====
        # Conv Block 1
        Conv1D(conv_filters[0], kernel_sizes[0], activation='relu', 
               input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout_rates[0]),
        
        # Conv Block 2
        Conv1D(conv_filters[1], kernel_sizes[1], activation='relu', 
               padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout_rates[1]),
        
        # Conv Block 3
        Conv1D(conv_filters[2], kernel_sizes[2], activation='relu', 
               padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout_rates[2]),
        
        # ===== LSTM TEMPORAL PROCESSING =====
        # After 3 maxpooling (2x each): 1440 -> 720 -> 360 -> 180 timesteps
        LSTM(lstm_units[0], return_sequences=True),
        BatchNormalization(),
        Dropout(dropout_rates[3]),
        
        LSTM(lstm_units[1], return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rates[4]),
        
        # ===== CLASSIFICATION HEAD =====
        Dense(dense_units[0], activation='relu'),
        Dropout(dropout_rates[5]),
        Dense(dense_units[1], activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    print("✅ Sequential Hybrid CNN-LSTM created successfully!")
    return model

def create_simplified_parallel_model(input_shape, num_classes=2, config=None):
    """
    Simplified parallel model for when you have more memory
    """
    if config is None:
        config = {}
    
    print("🧠 Creating Simplified Parallel Hybrid CNN-LSTM...")
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN Branch (simplified)
    cnn = Conv1D(32, 5, activation='relu', padding='same')(inputs)
    cnn = MaxPooling1D(4)(cnn)  # More aggressive pooling
    cnn = Conv1D(64, 3, activation='relu', padding='same')(cnn)
    cnn = MaxPooling1D(4)(cnn)
    cnn = GlobalAveragePooling1D()(cnn)
    
    # LSTM Branch (simplified)
    lstm = LSTM(32, return_sequences=False)(inputs)
    
    # Feature Fusion
    merged = Concatenate()([cnn, lstm])
    
    # Classification
    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    print("✅ Simplified Parallel Hybrid CNN-LSTM created successfully!")
    return model

def create_ultra_light_model(input_shape, num_classes=2):
    """
    Ultra-light model for very limited memory
    """
    print("🧠 Creating Ultra-Light Hybrid Model...")
    
    model = Sequential([
        # Simplified CNN
        Conv1D(16, 5, activation='relu', input_shape=input_shape),
        MaxPooling1D(4),
        Dropout(0.2),
        
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(4),
        Dropout(0.2),
        
        # Simplified LSTM
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        
        # Classification
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    print("✅ Ultra-Light Hybrid Model created successfully!")
    return model

def compile_model(model, learning_rate=0.001, metrics=None):
    """
    Model compilation with memory optimization
    """
    if metrics is None:
        metrics = ['accuracy']
    
    # Optimizer with gradient clipping for stability
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipvalue=1.0  # Gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    
    print("✅ Model compiled successfully!")
    return model