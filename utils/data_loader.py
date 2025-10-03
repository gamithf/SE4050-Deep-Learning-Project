import numpy as np
import joblib
from sklearn.utils.class_weight import compute_class_weight

class DataLoader:
    def __init__(self, data_path='data/processed_data/'):
        self.data_path = data_path
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.class_weights = None
        
    def load_data(self):
        """Load preprocessed data"""
        self.X_train = np.load(f'{self.data_path}/X_train.npy')
        self.X_val = np.load(f'{self.data_path}/X_val.npy')
        self.X_test = np.load(f'{self.data_path}/X_test.npy')
        self.y_train = np.load(f'{self.data_path}/y_train.npy')
        self.y_val = np.load(f'{self.data_path}/y_val.npy')
        self.y_test = np.load(f'{self.data_path}/y_test.npy')
        
        print(f"Data loaded successfully:")
        print(f"Train: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Val:   {self.X_val.shape}, {self.y_val.shape}")
        print(f"Test:  {self.X_test.shape}, {self.y_test.shape}")
        
        return self
    
    def get_class_weights(self):
        """Compute class weights for imbalance"""
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weight_dict = {0: self.class_weights[0], 1: self.class_weights[1]}
        print(f"Class weights: {class_weight_dict}")
        return class_weight_dict
    
    def get_data_shapes(self):
        """Return data shapes for model configuration"""
        return {
            'timesteps': self.X_train.shape[1],
            'features': self.X_train.shape[2],
            'num_classes': len(np.unique(self.y_train))
        }