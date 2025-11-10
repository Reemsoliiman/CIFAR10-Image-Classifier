import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model, test_dataset):
    """Evaluate the model and print metrics."""
    # Get predictions and true labels
    y_pred = []
    y_true = []
    for x, y in test_dataset:
        pred = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(pred, axis=1))
        y_true.extend(np.argmax(y, axis=1))
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]))
    
    # Compute test accuracy
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")