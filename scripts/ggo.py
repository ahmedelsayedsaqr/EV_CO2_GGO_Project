import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from scripts.data_prep import get_data_splits, get_preprocessor

# --- GGO Parameters (Simplified) ---
POPULATION_SIZE = 10
MAX_ITERATIONS = 10
LOWER_BOUND = [0.0001, 10, 10]  # [learning_rate_init, hidden_layer_size_1, hidden_layer_size_2]
UPPER_BOUND = [0.1, 100, 100]   # [learning_rate_init, hidden_layer_size_1, hidden_layer_size_2]

class GreylagGooseOptimizer:
    def __init__(self, population_size=POPULATION_SIZE, max_iterations=MAX_ITERATIONS, lb=LOWER_BOUND, ub=UPPER_BOUND):
        self.pop_size = population_size
        self.max_iter = max_iterations
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = len(lb)
        self.X_train, _, self.y_train, _ = get_data_splits()
        self.preprocessor = get_preprocessor()
        
        # Initialize population (geese positions)
        self.population = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        self.fitness = np.zeros(self.pop_size)
        self.best_goose_pos = np.zeros(self.dim)
        self.best_goose_fit = float('inf')

    def objective_function(self, params):
        """
        Objective function: Train MLP with given hyperparameters and return cross-validation RMSE.
        params: [learning_rate_init, hidden_layer_size_1, hidden_layer_size_2]
        """
        lr = params[0]
        hls1 = int(params[1])
        hls2 = int(params[2])
        
        # Ensure integer sizes are positive
        if hls1 <= 0 or hls2 <= 0:
            return float('inf')

        try:
            mlp = MLPRegressor(
                hidden_layer_sizes=(hls1, hls2),
                learning_rate_init=lr,
                max_iter=100, # Reduced for faster optimization
                random_state=42,
                solver='adam',
                tol=1e-3,
                n_iter_no_change=10
            )
            
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                       ('regressor', mlp)])
            
            # Use negative RMSE as scoring (sklearn maximizes score)
            scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                                     cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
            
            # We want to minimize RMSE, so we maximize the negative RMSE
            avg_neg_rmse = np.mean(scores)
            return -avg_neg_rmse # Return positive RMSE for clarity in optimization
        
        except Exception as e:
            # Handle potential errors during training (e.g., convergence issues)
            return float('inf')

    def optimize(self):
        """Main GGO optimization loop (simplified)."""
        print("Starting GGO Hyperparameter Optimization...")
        
        for i in range(self.pop_size):
            self.fitness[i] = self.objective_function(self.population[i])
            if self.fitness[i] < self.best_goose_fit:
                self.best_goose_fit = self.fitness[i]
                self.best_goose_pos = self.population[i].copy()

        for t in range(self.max_iter):
            # Simplified GGO update rule (inspired by PSO/GWO)
            # Geese move towards the best position found so far
            for i in range(self.pop_size):
                # Random factor for exploration
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Simplified velocity/step calculation
                step = r1 * (self.best_goose_pos - self.population[i]) + r2 * (self.population[i] - np.mean(self.population, axis=0))
                
                # Update position
                self.population[i] = self.population[i] + 0.1 * step # 0.1 is a simplified scaling factor
                
                # Apply bounds
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                
                # Evaluate new position
                new_fit = self.objective_function(self.population[i])
                self.fitness[i] = new_fit
                
                # Update global best
                if new_fit < self.best_goose_fit:
                    self.best_goose_fit = new_fit
                    self.best_goose_pos = self.population[i].copy()

            print(f"Iteration {t+1}/{self.max_iter}: Best RMSE = {self.best_goose_fit:.4f}")

        # Final best parameters
        best_params = {
            'learning_rate_init': self.best_goose_pos[0],
            'hidden_layer_size_1': int(self.best_goose_pos[1]),
            'hidden_layer_size_2': int(self.best_goose_pos[2]),
            'best_rmse': self.best_goose_fit
        }
        
        print("\nGGO Optimization Complete.")
        return best_params

if __name__ == '__main__':
    # Example run (will take some time)
    optimizer = GreylagGooseOptimizer(population_size=5, max_iterations=3)
    best_hps = optimizer.optimize()
    print("Best Hyperparameters found by GGO:")
    print(best_hps)
