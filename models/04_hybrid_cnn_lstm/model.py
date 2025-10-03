# models/04_hybrid_cnn_lstm/model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    BatchNormalization, Concatenate, GlobalMaxPooling1D, 
    GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2

def create_hybrid_model(input_shape, num_classes=2, config=None):
    """
    Create Advanced Hybrid CNN-LSTM model for depression classification
    """
    # Safe configuration with defaults
    if config is None:
        config = {}
    
    # Set default values with safe dictionary access
    conv_filters = config.get('conv_filters', [64, 128, 256])
    kernel_sizes = config.get('kernel_sizes', [7, 5, 3])
    lstm_units = config.get('lstm_units', [128, 64])
    dense_units = config.get('dense_units', [512, 256, 128])
    dropout_rates = config.get('dropout_rates', [0.2, 0.2, 0.3, 0.3, 0.5, 0.5, 0.3])
    l2_reg = config.get('l2_reg', 0.001)  # Safe access with default
    
    print(f"Input shape: {input_shape}")
    
    # Input layer
    inputs = Input(shape=input_shape, name='input_layer')
    
    # ===== PARALLEL BRANCH 1: CNN FOR LOCAL PATTERN EXTRACTION =====
    print("Building CNN branch...")
    
    # Main CNN branch
    conv1 = Conv1D(
        filters=conv_filters[0],
        kernel_size=kernel_sizes[0],
        activation='relu',
        padding='same',
        kernel_regularizer=l2(l2_reg),
        name='conv1'
    )(inputs)
    conv1 = BatchNormalization(name='bn_conv1')(conv1)
    conv1 = MaxPooling1D(pool_size=2, name='pool1')(conv1)
    conv1 = Dropout(dropout_rates[0], name='dropout_conv1')(conv1)
    
    conv2 = Conv1D(
        filters=conv_filters[1],
        kernel_size=kernel_sizes[1],
        activation='relu',
        padding='same',
        kernel_regularizer=l2(l2_reg),
        name='conv2'
    )(conv1)
    conv2 = BatchNormalization(name='bn_conv2')(conv2)
    conv2 = MaxPooling1D(pool_size=2, name='pool2')(conv2)
    conv2 = Dropout(dropout_rates[1], name='dropout_conv2')(conv2)
    
    conv3 = Conv1D(
        filters=conv_filters[2],
        kernel_size=kernel_sizes[2],
        activation='relu',
        padding='same',
        kernel_regularizer=l2(l2_reg),
        name='conv3'
    )(conv2)
    conv3 = BatchNormalization(name='bn_conv3')(conv3)
    conv3 = MaxPooling1D(pool_size=2, name='pool3')(conv3)
    conv3 = Dropout(dropout_rates[2], name='dropout_conv3')(conv3)
    
    # CNN Feature extraction
    cnn_max_pool = GlobalMaxPooling1D(name='cnn_global_max')(conv3)
    cnn_avg_pool = GlobalAveragePooling1D(name='cnn_global_avg')(conv3)
    
    # ===== PARALLEL BRANCH 2: LSTM FOR TEMPORAL DEPENDENCIES =====
    print("Building LSTM branch...")
    
    # LSTM layers
    lstm1 = LSTM(
        units=lstm_units[0],
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        name='lstm1'
    )(inputs)
    lstm1 = BatchNormalization(name='bn_lstm1')(lstm1)
    lstm1 = Dropout(dropout_rates[3], name='dropout_lstm1')(lstm1)
    
    lstm2 = LSTM(
        units=lstm_units[1],
        return_sequences=False,
        kernel_regularizer=l2(l2_reg),
        name='lstm2'
    )(lstm1)
    lstm2 = BatchNormalization(name='bn_lstm2')(lstm2)
    lstm2 = Dropout(dropout_rates[4], name='dropout_lstm2')(lstm2)
    
    # ===== FEATURE FUSION =====
    print("Merging features...")
    
    # Combine all features
    merged_features = Concatenate(name='feature_fusion')([
        cnn_max_pool,      # CNN max features
        cnn_avg_pool,      # CNN average features  
        lstm2              # LSTM temporal features
    ])
    
    print(f"Merged feature dimension: {merged_features.shape}")
    
    # ===== CLASSIFICATION HEAD =====
    print("Building classification head...")
    
    # Feature transformation layers
    x = Dense(
        dense_units[0],
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='dense1'
    )(merged_features)
    x = BatchNormalization(name='bn_dense1')(x)
    x = Dropout(dropout_rates[5], name='dropout_dense1')(x)
    
    # Intermediate layer
    x = Dense(
        dense_units[1],
        activation='relu',
        name='dense2'
    )(x)
    x = BatchNormalization(name='bn_dense2')(x)
    x = Dropout(dropout_rates[6], name='dropout_dense2')(x)
    
    # Final layer
    x = Dense(
        dense_units[2],
        activation='relu',
        name='dense3'
    )(x)
    
    # Output layer
    outputs = Dense(
        num_classes, 
        activation='softmax',
        name='output_layer'
    )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='Hybrid_CNN_LSTM')
    
    print("✅ Hybrid CNN-LSTM model created successfully!")
    return model

def compile_model(model, learning_rate=0.001, metrics=None):
    """
    Model compilation with safe metric handling
    """
    if metrics is None:
        metrics = ['accuracy']  # Default to just accuracy
    
    # Custom optimizer configuration
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    
    print("✅ Model compiled successfully!")
    return model

# Simplified sequential hybrid model
def create_sequential_hybrid_model(input_shape, num_classes=2, config=None):
    """
    Sequential Hybrid CNN-LSTM (Simpler alternative)
    """
    from tensorflow.keras.models import Sequential
    
    # Safe configuration with defaults
    if config is None:
        config = {}
    
    conv_filters = config.get('conv_filters', [64, 128, 256])
    kernel_sizes = config.get('kernel_sizes', [7, 5, 3])
    lstm_units = config.get('lstm_units', [128, 64])
    dense_units = config.get('dense_units', [256, 128])
    dropout_rates = config.get('dropout_rates', [0.3, 0.3, 0.3, 0.5, 0.5])
    
    model = Sequential([
        # CNN Feature Extraction
        Conv1D(conv_filters[0], kernel_sizes[0], activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout_rates[0]),
        
        Conv1D(conv_filters[1], kernel_sizes[1], activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout_rates[1]),
        
        Conv1D(conv_filters[2], kernel_sizes[2], activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(dropout_rates[2]),
        
        # LSTM Temporal Processing
        LSTM(lstm_units[0], return_sequences=True),
        BatchNormalization(),
        Dropout(dropout_rates[3]),
        
        LSTM(lstm_units[1], return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rates[4]),
        
        # Classification Head
        Dense(dense_units[0], activation='relu'),
        Dropout(0.5),
        Dense(dense_units[1], activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    return model