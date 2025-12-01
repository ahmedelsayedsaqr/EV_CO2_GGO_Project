import json
from scripts.ggo import GreylagGooseOptimizer

def run_optimization():
    """Runs the GGO optimization and saves the best hyperparameters."""
    print("--- Starting Hyperparameter Optimization with GGO ---")
    
    # Initialize and run the optimizer
    # Using small population and iterations for a quick demonstration
    optimizer = GreylagGooseOptimizer(population_size=10, max_iterations=10)
    best_params = optimizer.optimize()
    
    # Save the best parameters to a JSON file
    output_path = '../results/best_params.json'
    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print(f"\nOptimization complete. Best parameters saved to {output_path}")
    print(json.dumps(best_params, indent=4))
    
    return best_params

if __name__ == '__main__':
    run_optimization()
