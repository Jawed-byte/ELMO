# ELMO 

This project implements and compares different ELMO (Embeddings from Language Models) configurations against traditional embedding methods like SVD and Skip-gram. We explore two main ELMO variants: one with trainable lambda parameters and another with a learnable function.

## Models Implemented

### 1. ELMO with Trainable Lambda
- Randomly initialized lambda parameters
- Frozen LSTM weights
- Learning rate: 0.001
- Epochs: 5

### 2. ELMO with Learnable Function
- Replaced lambda with non-linear function
- Context learning through function adaptation
- Same basic configuration as trainable lambda

### 3. Comparison Models
- SVD (Window size = 3)
- Skip-gram (Window size = 5)

## Results

### ELMO - Trainable Lambda

#### Test Data Performance
- Overall Accuracy: 0.91
- Macro and Weighted Averages: 0.91 (Precision, Recall, F1-score)
- Balanced performance across all classes

#### Training Data Performance
- Overall Accuracy: 0.94
- Class 1 highest F1-score: 0.98
- Class 3 lowest F1-score: 0.91

### ELMO - Learnable Function

#### Test Data Performance
- Precision Range: 0.85-0.96
- Recall Range: 0.83-0.97
- Overall Accuracy: 0.90

#### Training Data Performance
- Overall Accuracy: 0.91
- Strong performance in Class 1
- Slightly lower precision in Classes 2 and 3

## Model Comparison

### SVD (Window Size = 3)
- Train Accuracy: 0.7980
- Test Accuracy: 0.7812
- Test F1 Score: 0.7817

### Skip-gram (Window Size = 5)
- Train Accuracy: 0.9838
- Test Accuracy: 0.8637
- Test F1 Score: 0.8637

### ELMO (Best Configuration)
- Train Accuracy: 0.94
- Test Accuracy: 0.91
- Best overall performance

