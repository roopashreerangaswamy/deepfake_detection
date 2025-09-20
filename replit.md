# Replit Project Configuration

## Overview

This is a deepfake detection application built with Streamlit that uses deep learning to identify manipulated images. The system employs a Convolutional Neural Network (CNN) architecture with Grad-CAM (Gradient-weighted Class Activation Mapping) visualization to provide explainable AI predictions, helping users understand which parts of an image influenced the deepfake classification decision.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Interface**: Single-page web application with image upload capabilities
- **Visualization**: Integrated Grad-CAM heatmap overlays for prediction explanations
- **User Flow**: Upload image → Process → Display prediction with visual explanation

### Machine Learning Architecture
- **Model Type**: Convolutional Neural Network (CNN) for binary classification
- **Architecture**: Sequential CNN with progressive feature extraction
  - Conv2D layers (32 → 64 → 128 filters) with ReLU activation
  - MaxPooling2D layers for dimensionality reduction
  - Dense layers with dropout for classification
  - Sigmoid output for binary deepfake/real classification
- **Input Processing**: Images resized to 128x128 pixels, normalized for model input
- **Explainability**: Grad-CAM integration using the last convolutional layer for attention visualization

### Data Processing
- **Image Handling**: PIL (Python Imaging Library) for image loading and preprocessing
- **Computer Vision**: OpenCV for advanced image processing operations (optional dependency)
- **Input Validation**: Automatic image format conversion and size standardization

### Model Management
- **Storage**: Keras model saved as `deepfake_detector.keras` file
- **Loading**: Dynamic model loading with error handling
- **Architecture Extraction**: Separate model and feature layer references for Grad-CAM

### Error Handling & Dependencies
- **Graceful Degradation**: Optional dependency handling for TensorFlow and OpenCV
- **Fallback Mechanisms**: Application continues with reduced functionality if dependencies missing
- **User Feedback**: Clear error messages for missing dependencies or model files

## External Dependencies

### Core ML Framework
- **TensorFlow/Keras**: Primary deep learning framework for model training and inference
- **NumPy**: Numerical computing for array operations and data manipulation

### Image Processing
- **PIL (Pillow)**: Image loading, conversion, and basic preprocessing
- **OpenCV**: Advanced computer vision operations and image manipulations (optional)

### Web Framework
- **Streamlit**: Web application framework for creating the user interface and handling file uploads

### Model Storage
- **Local File System**: Keras model persistence in `.keras` format
- **No External Model APIs**: Self-contained model deployment without cloud dependencies