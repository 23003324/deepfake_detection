
# Deepfake Image Detection using CNN (K-Fold) and MobileNetV2
## Project Overview
```
This project focuses on deepfake image detection using deep learning techniques. With the rapid growth of manipulated media, identifying fake images has become a critical challenge. In this work, two lightweight and practical models are implemented and evaluated:

A custom Convolutional Neural Network (CNN) trained using K-Fold Cross Validation

A MobileNetV2 model using transfer learning

The project prioritizes simplicity, reproducibility, and low computational cost, making it suitable for academic and resource-constrained environments.
```

## Objectives
```
To detect whether a facial image is real or deepfake

To compare a custom CNN (with K-Fold validation) against MobileNetV2

To analyze model behavior using Grad-CAM visualization

To evaluate performance under limited hardware constraints
```
## Models Used

### 1. CNN with K-Fold Cross Validation
Custom CNN architecture

K-Fold Cross Validation to reduce bias and variance

Helps in obtaining a more reliable performance estimate

### 2.MobileNetV2
Pretrained on ImageNet

Fine-tuned for binary classification (Real vs Fake)

Chosen for its lightweight and efficient design

## Results
Accuracy achieved: ~70–71%

Performance is moderate due to:

High visual similarity between real and fake images

Compression artifacts in datasets

Limited dataset size

Despite lower accuracy, the models demonstrate feasibility and consistency.

## Explainability
Grad-CAM is used to visualize important facial regions

Helps understand where the model focuses while making predictions

Improves interpretability of CNN and MobileNetV2 decisions

## Technologies Used
Python

TensorFlow / Keras

NumPy, OpenCV

Grad-CAM

Google Colab / Local GPU

📁 Project Structure ├── cnn_kfold.ipynb ├── mobilenetv2.ipynb ├── gradcam_visualization.ipynb ├── dataset/ │ ├── real/ │ └── fake/ └── README.md

## Limitations
Accuracy is limited (~70–71%)

Sensitive to image quality and compression

Does not handle video-level deepfake detection

## Future Improvements
Ensemble learning (CNN + MobileNetV2)

Larger and more diverse datasets

Video-based deepfake detection

Improved preprocessing and face alignment

## Conclusion
This project demonstrates that lightweight deep learning models can be used for deepfake image detection with reasonable performance. While not production-ready, it serves as a strong academic and experimental baseline for further research.
