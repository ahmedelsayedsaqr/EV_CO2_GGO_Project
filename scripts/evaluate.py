import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scripts.train_mlp import train_final_model

def evaluate_model(pipeline, X_test, y_test):
    """Evaluates the trained model and saves metrics and figures."""
    
    # 1. Prediction
    y_pred = pipeline.predict(X_test)
    
    # 2. Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "MSE": float(f"{mse:.4f}"),
        "RMSE": float(f"{rmse:.4f}"),
        "MAE": float(f"{mae:.4f}"),
        "R2": float(f"{r2:.4f}")
    }
    
    # 3. Save Metrics
    metrics_path = '../results/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("\n--- Model Performance Metrics ---")
    print(json.dumps(metrics, indent=4))
    print(f"Metrics saved to {metrics_path}")
    
    # 4. Create Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    
    # Add ideal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
    
    plt.title('Actual vs. Predicted CO2 Emissions (GGO-Optimized MLP)', fontsize=14)
    plt.xlabel('Actual CO2 Emissions (g/km)', fontsize=12)
    plt.ylabel('Predicted CO2 Emissions (g/km)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    figure_path = '../results/figures/actual_vs_predicted.png'
    plt.savefig(figure_path)
    print(f"Scatter plot saved as {figure_path}")
    
    return metrics

if __name__ == '__main__':
    # Run optimization (if not run), train model, and evaluate
    pipeline, X_test, y_test = train_final_model()
    evaluate_model(pipeline, X_test, y_test)
