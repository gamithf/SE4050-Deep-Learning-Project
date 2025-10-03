import sys
import os
sys.path.append('../../')

from utils.data_loader import DataLoader
from utils.common import create_callbacks, setup_gpu
from utils.metrics import ModelEvaluator
from model import create_cnn_model, compile_model

def train_cnn_model():
    """Training script - COMMON structure for all models"""
    # Setup
    setup_gpu()
    model_name = "CNN_Model"
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader()
    data_loader.load_data()
    data_shapes = data_loader.get_data_shapes()
    class_weights = data_loader.get_class_weights()
    
    # Create model
    print("Creating model...")
    model = create_cnn_model(
        input_shape=(data_shapes['timesteps'], data_shapes['features'])
    )
    model = compile_model(model)
    
    # Display model architecture
    model.summary()
    
    # Train model
    print("Training model...")
    callbacks = create_callbacks(model_name)
    
    history = model.fit(
        data_loader.X_train, data_loader.y_train,
        validation_data=(data_loader.X_val, data_loader.y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(model_name)
    evaluator.set_history(history)
    
    # Make predictions
    y_pred_proba = model.predict(data_loader.X_test)
    y_pred = y_pred_proba.argmax(axis=1)
    
    evaluator.set_predictions(data_loader.y_test, y_pred, y_pred_proba[:, 1])
    
    # Generate and save results
    report, cm = evaluator.generate_classification_report()
    
    # Plot results
    evaluator.plot_training_history(f'results/model_performance/{model_name}_training.png')
    evaluator.plot_confusion_matrix(cm, f'results/model_performance/{model_name}_cm.png')
    evaluator.plot_roc_curve(f'results/model_performance/{model_name}_roc.png')
    evaluator.save_results(report, cm)
    
    print(f"\n=== {model_name} Results ===")
    print(f"Test Accuracy: {report['accuracy']:.3f}")
    print(f"Sensitivity: {report['sensitivity']:.3f}")
    print(f"Specificity: {report['specificity']:.3f}")
    if 'roc_auc' in report:
        print(f"ROC AUC: {report['roc_auc']:.3f}")
    
    return model, history, report

if __name__ == "__main__":
    train_cnn_model()