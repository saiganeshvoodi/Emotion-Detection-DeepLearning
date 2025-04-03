# Emotion-Detection-DeepLearning 

A deep learning-based Facial Expression Recognition (FER) system using Convolutional Neural Networks (CNNs) to classify emotions from facial images. This pre-trained model detects **7 emotions**: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral. Includes a Jupyter Notebook for running predictions and real-time emotion detection.


## Key Features
- **Pre-trained CNN Model**: A CNN architecture trained on the FER2013 dataset (saved as `model.h5`).
- **Real-Time Detection**: Jupyter Notebook script for live emotion prediction using a webcam.
- **Easy Setup**: Load the model and run inference directly with provided code.


## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Emotion-Detection-DeepLearning.git
   cd Emotion-Detection-DeepLearning
   ```
## Dependencies
The project requires the following Python libraries:
- **TensorFlow/Keras**: For building, training, and deploying the CNN model.
- **OpenCV**: For real-time image/video processing and face detection.
- **NumPy & Pandas**: For numerical computations and data handling.
- **Matplotlib & Seaborn**: For visualizing training metrics and confusion matrices.
- **Scikit-learn**: For generating classification reports and evaluation metrics.
- **Jupyter**: For running the provided `.ipynb` notebook.

Install all dependencies using:
```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Usage
### Run the Jupyter Notebook
Open the provided `realfeed.ipynb` file to:
   - Load the pre-trained model (`model.h5`).
   - Predict emotions from static images or use your webcam for real-time detection.

### Real-Time Detection
The notebook includes code to:
1. Access your webcam.
2. Detect faces and classify emotions live.

## Dataset
The model is trained on the **FER2013 dataset**:
- **Source**: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **Details**: 35,887 grayscale images (48x48 pixels) labeled with 7 emotions.
- **Preprocessing**: Normalization and augmentation applied during training.

## Model Architecture
The pre-trained CNN (`model.h5`) uses:
- **5 Convolutional Blocks**: Each with zero-padding, ReLU activation, and max-pooling.
- **Dense Layers**: Two fully connected layers (4,096 units) with dropout for regularization.
- **Output Layer**: Softmax activation for classification into 7 emotions.

## Results
- Achieved **~65% test accuracy** on the FER2013 dataset.
- Optimized for real-time performance with minimal lag during inference.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- Dataset: [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
