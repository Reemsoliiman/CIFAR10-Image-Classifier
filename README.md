# CIFAR10-Image-Classifier

This project deploys a dense neural network to classify CIFAR-10 images (60,000 32x32 color images across 10 classes) using TensorFlow. Images are flattened (no CNN) and fed into a fully connected network. Includes data preprocessing, model training, evaluation, and a Streamlit app for inference.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cifar10-classifier.git
   cd cifar10-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python src/models/model.py --config config/config.yaml
   ```
4. Launch the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```
5. Run tests:
   ```bash
   python -m unittest tests/test_model.py
   ```

## Structure
- `src/`: Source code for data loading, model training, utilities, and Streamlit app.
- `config/`: Hyperparameters in `config.yaml`.
- `models/`: Saved model checkpoints (e.g., `cifar10_model.h5`).
- `notebooks/`: Jupyter notebook for data exploration.
- `tests/`: Unit tests for data and model.
- `data/`: Script to download CIFAR-10 (optional).

## Requirements
See `requirements.txt` for dependencies.