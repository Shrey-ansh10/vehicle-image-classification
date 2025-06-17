# AutoVision: Multi-Class Vehicle Recognition with CNNs

## 📝 Project Overview
AutoVision is a deep learning-based computer vision project that focuses on accurate multi-class vehicle classification using Convolutional Neural Networks (CNNs). The project implements state-of-the-art deep learning techniques to classify different types of vehicles from images, demonstrating the practical application of computer vision in automotive and transportation sectors.

## 🎯 Objectives
- Develop a robust CNN model for multi-class vehicle classification
- Implement and compare different learning rate strategies
- Optimize model performance through data augmentation
- Evaluate model performance using various metrics
- Create a scalable and maintainable deep learning pipeline

## 🛠️ Technologies Used
- **Python** - Primary programming language
- **TensorFlow & Keras** - Deep learning framework
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **Matplotlib & Seaborn** - Data visualization
- **scikit-learn** - Model evaluation metrics
- **OpenCV** - Image processing operations

## 📊 Methodology

### 1. Data Preprocessing
- **Image Loading**: Implemented systematic loading of vehicle images from organized class directories
- **Data Augmentation**: Applied multiple augmentation techniques:
  - Rotation (±20 degrees)
  - Width and height shifting
  - Brightness adjustment
  - Zoom variation
  - Horizontal flipping
  - Rescaling pixel values to [0,1]

### 2. Model Architecture
The CNN architecture consists of multiple convolutional and pooling layers followed by dense layers:
```
- Convolutional layers with increasing filter sizes
- MaxPooling layers for spatial dimension reduction
- Dropout layers for regularization
- Dense layers for final classification
- Softmax activation for multi-class output
```

### 3. Training Strategy
- **Batch Size**: Optimized for memory efficiency and training speed
- **Learning Rate Management**:
  - Implemented Exponential Decay strategy
  - Tested ReduceLROnPlateau for adaptive learning rate adjustment
- **Early Stopping**: Prevented overfitting by monitoring validation metrics

### 4. Model Evaluation
Comprehensive evaluation using multiple metrics:
- Accuracy and Loss curves
- Precision, Recall, and F1-Score
- Confusion Matrix
- Classification Report
- Cross-validation scores

## 📈 Results
- Successfully implemented a multi-class vehicle classification system
- Achieved significant accuracy in vehicle type prediction
- Demonstrated effective handling of overfitting through regularization techniques
- Established optimal learning rate strategies for model convergence

## 🔍 Key Findings
1. Data augmentation significantly improved model generalization
2. Learning rate decay strategies showed notable impact on training stability
3. Model demonstrated robust performance across different vehicle classes
4. Dropout layers effectively prevented overfitting

## 🚀 Future Improvements
- Integration of transfer learning using pre-trained models (ResNet, VGG)
- Implementation of more sophisticated data augmentation techniques
- Exploration of model compression for mobile deployment
- Addition of real-time classification capabilities
- Integration with object detection for complete vehicle analysis

## 📦 Project Structure
```
vehicle-image-classification/
│
├── Image Classification Project.ipynb    # Main project notebook
└── README.md                            # Project documentation
```

## 🎓 Lessons Learned
- Importance of systematic data preprocessing in computer vision
- Impact of learning rate strategies on model convergence
- Value of comprehensive model evaluation metrics
- Significance of data augmentation in improving model robustness

## 📫 Contact
For any queries regarding this project, please feel free to raise an issue in the repository.