import unittest
import tensorflow as tf
import numpy as np
from src.data.dataloader import load_cifar10_data
from src.models.model import build_model

class TestCIFAR10Classifier(unittest.TestCase):
    def test_data_loading(self):
        """Test if CIFAR-10 data loads correctly."""
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()
        self.assertEqual(x_train.shape, (50000, 32 * 32 * 3))
        self.assertEqual(y_train.shape, (50000, 10))
        self.assertEqual(x_test.shape, (10000, 32 * 32 * 3))
        self.assertEqual(y_test.shape, (10000, 10))
    
    def test_model_output(self):
        """Test if model outputs correct shape."""
        model = build_model()
        x = np.random.rand(1, 32 * 32 * 3).astype('float32')
        pred = model.predict(x)
        self.assertEqual(pred.shape, (1, 10))

if __name__ == '__main__':
    unittest.main()