import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from scripts.data_prep import get_data_splits, get_preprocessor
from scripts.optimize import run_optimization

def train_final_model(best_params=None):
    """Trains the final MLP model using the best hyperparameters found by GGO."""
    
    # 1. Load best parameters if not provided
    if best_params is None:
        try:
            with open('../results/best_params.json', 'r') as f:
                best_params = json.load(f)
        except FileNotFoundError:
            print("Best parameters not found. Running optimization...")
            best_params = run_optimization()
        except Exception as e:
            print(f"Error loading best parameters: {e}. Using default parameters.")
            best_params = {
                'learning_rate_init': 0.001,
                'hidden_layer_size_1': 100,
                'hidden_layer_size_2': 50
            }

    # 2. Prepare data
    X_train, X_test, y_train, y_test = get_data_splits()
    preprocessor = get_preprocessor()

    # 3. Configure MLP with optimized parameters
    hls1 = int(best_params.get('hidden_layer_size_1', 100))
    hls2 = int(best_params.get('hidden_layer_size_2', 50))
    lr = best_params.get('learning_rate_init', 0.001)
    
    print(f"\n--- Training Final MLP Model with Optimized HPs: ({hls1}, {hls2}), LR={lr:.5f} ---")

    mlp = MLPRegressor(
        hidden_layer_sizes=(hls1, hls2),
        learning_rate_init=lr,
        max_iter=500, # Increased max_iter for final training
        random_state=42,
        solver='adam',
        tol=1e-4,
        n_iter_no_change=20
    )

    # 4. Create and train pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', mlp)])
    
    pipeline.fit(X_train, y_train)
    print("Final model training complete.")
    
    return pipeline, X_test, y_test

if __name__ == '__main__':
    # This script is primarily for training, evaluation is in evaluate.py
    pipeline, X_test, y_test = train_final_model()
    print(f"Model trained and ready for evaluation on {len(X_test)} test samples.")
