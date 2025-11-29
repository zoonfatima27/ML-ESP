# -----------------------------
# Basic Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# -----------------------------
# For reproducibility
# -----------------------------
np.random.seed(123)
torch.manual_seed(123)

# -----------------------------
# Load Data
# -----------------------------
data_path = "Z:/OneDrive - National University of Sciences & Technology/Desktop/ML Project/insurance.csv"
data = pd.read_csv(data_path)
print("Dataset shape:", data.shape)
print(data.head())

# -----------------------------
# Preprocessing Functions
# -----------------------------
def binary_encode(df, column, positive_value):
    df = df.copy()
    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1).drop(column, axis=1)
    return df

def preprocess_inputs(df, scaler, train_size=0.7):
    df = df.copy()
    df = binary_encode(df, 'sex', 'male')
    df = binary_encode(df, 'smoker', 'yes')
    df = onehot_encode(df, 'children', 'ch')
    df = onehot_encode(df, 'region', 're')
    
    y = df['charges'].copy()
    X = df.drop('charges', axis=1).copy()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=123
    )
    return X_train, X_test, y_train, y_test

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = preprocess_inputs(data, StandardScaler(), train_size=0.7)

# -----------------------------
# Classical ML: Linear Regression
# -----------------------------
ls_model = LinearRegression()
ls_model.fit(X_train, y_train)
print(f"Linear Regression R² on test: {ls_model.score(X_test, y_test):.4f}")

# -----------------------------
# Ridge, Lasso, ElasticNet with GridSearchCV
# -----------------------------
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

param_grids = {
    'Ridge': {'alpha': [0.01, 0.1, 1, 10, 100]},
    'Lasso': {'alpha': [0.001, 0.01, 0.1, 1], 'max_iter': [20000]},
    'ElasticNet': {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.2, 0.5, 0.8], 'max_iter': [20000]}
}

best_models = {}
for name, model in [('Ridge', Ridge()), ('Lasso', Lasso()), ('ElasticNet', ElasticNet())]:
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"{name} best params: {grid.best_params_} | R² on test: {grid.score(X_test, y_test):.4f}")

# -----------------------------
# Neural Network (PyTorch)
# -----------------------------
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

val_size = int(0.2 * len(X_tensor))
train_size = len(X_tensor) - val_size
train_dataset, val_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class InsuranceNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

nn_model = InsuranceNN(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

best_val_loss = np.inf
patience = 10
counter = 0
n_epochs = 200

for epoch in range(n_epochs):
    nn_model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = nn_model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
    
    # Validation
    nn_model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            val_pred = nn_model(xb)
            val_loss += criterion(val_pred, yb).item() * len(yb)
    val_loss /= len(val_dataset)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(nn_model.state_dict(), 'best_nn_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

nn_model.load_state_dict(torch.load('best_nn_model.pth'))

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_pred_nn = nn_model(X_test_tensor).detach().numpy().flatten()

# -----------------------------
# Predictions & Metrics
# -----------------------------
predictions = {
    'Best Ridge': best_models['Ridge'].predict(X_test),
    'Best Lasso': best_models['Lasso'].predict(X_test),
    'Best ElasticNet': best_models['ElasticNet'].predict(X_test),
    'Neural Network': y_pred_nn
}

metrics = {}
for name, preds in predictions.items():
    r2 = np.round(np.corrcoef(y_test, preds)[0,1]**2, 4)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    metrics[name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae}

metrics_df = pd.DataFrame(metrics).T
print(metrics_df)

# -----------------------------
# Plots
# -----------------------------
# R² Comparison
plt.figure(figsize=(8,5))
plt.bar(metrics_df.index, metrics_df['R²'], color='skyblue')
plt.ylabel('R² Score')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.show()

# Scatter plots
plt.figure(figsize=(16, 12))
for i, (name, preds) in enumerate(predictions.items(), 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=y_test, y=preds)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title(f'{name}: Predicted vs Actual')
plt.tight_layout()
plt.show()
