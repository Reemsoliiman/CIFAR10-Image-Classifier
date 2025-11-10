import tensorflow as tf
from tensorflow.keras import layers, models
import yaml
import os
import sys

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.helpers import evaluate_model
from src.data.dataloader import get_data_loaders

def build_model(input_shape=(32 * 32 * 3,), num_classes=10):
    """Build a dense neural network for CIFAR-10."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(config_path):
    """Train the model using parameters from config.yaml."""
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load data
    train_dataset, test_dataset = get_data_loaders(config['batch_size'])
    
    # Build model
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=test_dataset
    )
    
    # Evaluate model
    evaluate_model(model, test_dataset)
    
    # Save model
    os.makedirs(config['model_save_path'], exist_ok=True)
    model.save(os.path.join(config['model_save_path'], 'cifar10_model.h5'))
    
    return model, history

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train CIFAR-10 classifier.')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    train_model(args.config)