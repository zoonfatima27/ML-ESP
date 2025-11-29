# ML-ESP
Medical Expense Prediction
# Medical Cost Prediction

## Team Members
- Zoonash Fatima

## Abstract
We predict medical charges based on patient demographics and medical history. Classical ML models and a Neural Network were implemented, with proper preprocessing and hyperparameter tuning.

## Introduction
Problem statement and objectives…

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
