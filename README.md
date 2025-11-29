# ML-ESP
Medical Expense Prediction
# Medical Cost Prediction

## Team Members
- Zoonash Fatima

## Abstract
We predict medical charges based on patient demographics and medical history. Classical ML models and a Neural Network were implemented, with proper preprocessing and hyperparameter tuning.

## Introduction

In this project, we aim to predict medical insurance charges using patient demographics and medical history. Understanding the factors affecting insurance premiums is crucial for both healthcare providers and clients.

**Problem Statement:**  
Medical insurance charges vary widely based on factors like age, BMI, smoking habits, and region. Accurately predicting these charges can help insurance companies price policies better and help individuals plan their finances.

**Objectives:**  
- Build models to predict insurance charges with high accuracy.  
- Compare classical machine learning models (Ridge, Lasso, ElasticNet) with a Neural Network.  
- Identify the most influential features affecting insurance costs.  
- Provide insights that can guide business decisions for insurance pricing.


## Dataset Description
- Source: Kaggle (link)
- Size: 1338 rows, 7 features
- Features: age, sex, bmi, children, smoker, region, charges
- Preprocessing: Encoding categorical features, scaling numerical features

## Methodology
### Classical ML Approaches
- Ridge, Lasso, ElasticNet
- Hyperparameter tuning via GridSearchCV

### Deep Learning
- Neural Network implemented in PyTorch
- Dropout regularization, early stopping

## Results & Analysis
### Performance Comparison
- See plots in `plots/` folder

### Visualizations
- Bar chart: R² scores comparison
- Scatter plots: Predicted vs Actual
- Feature importance (Ridge/Lasso/ElasticNet)

### Business Impact Analysis
- Most influential features: smoker, age, BMI
- NN gives higher accuracy; classical models provide interpretability

## Conclusion & Future Work
- Classical ML models work well after hyperparameter tuning
- NN slightly better but less interpretable
- Future work: add features like physical activity, medical history, regional health data, deeper NN, ensemble methods

## References
- [Kaggle Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance/data)
- [YouTube Video](https://www.youtube.com/watch?v=3GCv4Qq5DZQ)

