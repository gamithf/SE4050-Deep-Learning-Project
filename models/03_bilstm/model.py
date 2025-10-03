import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization

def create_bilstm_model(input_shape, num_classes=2):
    """
    Create Bidirectional LSTM model for depression classification
    Team Member: CHANGE ONLY THIS FUNCTION for BiLSTM-specific architecture
    """
    model = Sequential([
        # First Bidirectional LSTM layer
        Bidirectional(
            LSTM(128, return_sequences=True), 
            input_shape=input_shape,
            merge_mode='concat'
        ),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second Bidirectional LSTM layer
        Bidirectional(
            LSTM(64, return_sequences=True),
            merge_mode='concat'
        ),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third Bidirectional LSTM layer
        Bidirectional(
            LSTM(32, return_sequences=False),
            merge_mode='concat'
        ),
        BatchNormalization(),
        Dropout(0.3),
        
        # Classification Head
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    COMMON compile function - Same for all models
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model