# AutoVision: Multi-Class Vehicle Recognition with CNNs

## 📝 Project Overview
AutoVision is a deep learning-based computer vision project focused on multi-class vehicle classification using Convolutional Neural Networks (CNNs). The project implements a systematic approach to train and evaluate models for classifying 7 different types of vehicles, exploring various learning rate strategies and addressing critical challenges like overfitting and class imbalance.

## 🎯 Objectives
- Develop and evaluate CNN models for 7-class vehicle classification
- Compare two learning rate strategies: Exponential Decay and ReduceLROnPlateau
- Implement effective data augmentation and regularization techniques
- Address class imbalance using computed class weights
- Analyze model performance through comprehensive metrics and visualizations

## 🛠️ Technologies Used
- **Python** - Primary programming language
- **TensorFlow & Keras** - Deep learning framework and image processing
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization and performance plots
- **scikit-learn** - Model evaluation metrics and class weight computation

## 📊 Methodology

### 1. Data Preprocessing
- **Image Loading**: Implemented using TensorFlow/Keras' ImageDataGenerator for efficient batch processing
- **Data Augmentation**: Applied comprehensive augmentation techniques:
  - Rotation (±30 degrees)
  - Width and height shifting (30%)
  - Brightness adjustment (0.8-1.2 range)
  - Zoom variation (30%)
  - Horizontal flipping
  - Rescaling pixel values to [0,1]
- **Data Split**: 80-20 train-validation split using validation_split parameter

### 2. Model Architecture
Two CNN models were implemented with identical architectures but different learning rate strategies:
```
1. Input Layer (224x224x3)
2. Convolutional Layers:
   - Conv2D(32) + BatchNorm + MaxPool
   - Conv2D(64) + BatchNorm + MaxPool
   - Conv2D(128) + BatchNorm + MaxPool
3. Global Average Pooling
4. Dense Layer (128 units) with ReLU
5. Dropout (0.5) for regularization
6. Output Layer (7 units) with Softmax
```

### 3. Learning Rate Strategies
#### Model 1: Exponential Decay
- Initial learning rate: 1e-3
- Decay steps: 1000
- Decay rate: 0.96
- Training conducted in two phases (11 + 10 epochs)

#### Model 2: ReduceLROnPlateau
- Initial learning rate: 1e-3
- Reduction factor: 0.3
- Patience: 3 epochs
- Minimum learning rate: 1e-6
- Maximum epochs: 20

### 4. Training Features
- **Batch Size**: 32 for optimal memory usage
- **Class Weights**: Computed and applied to handle class imbalance
- **Early Stopping**: Monitored validation loss with patience=5
- **Best Weights Restoration**: Enabled for optimal model selection

## 📈 Results & Analysis

### Model 1 (Exponential Decay)
- Initial performance: ~71% test accuracy
- Extended training led to overfitting:
  - Decreasing validation accuracy
  - Increasing validation loss
  - Poor generalization on unseen data
- Confusion matrix showed inconsistent performance across classes

### Model 2 (ReduceLROnPlateau)
- Better generalization compared to Model 1
- More stable training metrics
- Adaptive learning rate helped prevent overfitting
- Achieved accuracy within the optimal range (70-85%)

## 🔍 Key Findings
1. Learning Rate Impact:
   - Exponential decay showed limitations in preventing overfitting
   - ReduceLROnPlateau provided better adaptation to training dynamics
2. Model Architecture Insights:
   - Base CNN architecture (3 conv layers) achieved reasonable performance
   - BatchNormalization and Dropout were crucial for training stability
3. Training Observations:
   - Class weights effectively addressed imbalance issues
   - Early stopping prevented significant overfitting
   - Model performance plateaued in the 70-85% accuracy range

## 🚀 Scope for Improvement
Based on experimental results, several approaches could enhance model performance:
- Implement transfer learning using pre-trained models (ResNet, VGG)
- Explore more advanced model architectures beyond the base 3-layer CNN
- Investigate additional regularization techniques
- Implement cross-validation for more robust evaluation
- Fine-tune hyperparameters more extensively

## 📦 Project Structure
```
vehicle-image-classification/
│
├── Image Classification Project.ipynb    # Main notebook with implementation and analysis
├── models/                              # Directory containing trained models
│   ├── vehicle_image_classification_model_1_v2.keras    # Model with Exponential Decay strategy
│   └── vehicle_image_classification_model_2.keras       # Model with ReduceLROnPlateau strategy
└── README.md                            # Project documentation
```

The `models/` directory contains two trained models:
- `model_1_v2`: Implements exponential decay learning rate strategy (trained for 11+10 epochs)
- `model_2`: Uses ReduceLROnPlateau for adaptive learning rate adjustment (trained for up to 20 epochs)

## 🎓 Key Learning Outcomes
1. Model Architecture:
   - Simple CNN architectures can achieve reasonable performance (70-85% accuracy)
   - Additional complexity might be needed for higher accuracy

2. Training Dynamics:
   - Different learning rate strategies significantly impact model convergence
   - ReduceLROnPlateau provides better adaptation than fixed decay schedules

3. Performance Optimization:
   - Class weights are essential for imbalanced datasets
   - Early stopping and learning rate adaptation prevent overfitting
   - Comprehensive evaluation metrics are crucial for understanding model behavior

## 📫 Contact
For any queries regarding this project, please feel free to raise an issue in the repository.