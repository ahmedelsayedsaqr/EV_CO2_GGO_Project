import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH = '../data/Fuel_Consumption_2000-2022.csv'

def load_data():
    """Loads the dataset and cleans column names."""
    df = pd.read_csv(DATA_PATH)
    # Rename columns based on initial inspection for clarity
    df.columns = [
        'Year', 'Make', 'Model', 'Vehicle_Class', 'Engine_Size_L', 'Cylinders',
        'Transmission', 'Fuel_Type', 'Fuel_City_L_100km', 'Fuel_Hwy_L_100km',
        'Fuel_Comb_L_100km', 'Fuel_Comb_mpg', 'CO2_Emissions_g_km'
    ]
    return df

def get_preprocessor():
    """Creates and returns the ColumnTransformer for preprocessing."""
    numerical_features = ['Engine_Size_L', 'Cylinders', 'Fuel_Comb_L_100km']
    categorical_features = ['Vehicle_Class', 'Transmission', 'Fuel_Type']

    # Create the preprocessing pipeline for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def get_data_splits(test_size=0.2, random_state=42):
    """Loads data, selects features, and splits into train/test sets."""
    df = load_data()
    
    features = [
        'Engine_Size_L', 'Cylinders', 'Fuel_Comb_L_100km',
        'Vehicle_Class', 'Transmission', 'Fuel_Type'
    ]
    target = 'CO2_Emissions_g_km'

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Example usage and verification
    X_train, X_test, y_train, y_test = get_data_splits()
    preprocessor = get_preprocessor()
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("Data preparation complete.")
    print(f"Training set size: {X_train_processed.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Number of features after One-Hot Encoding: {X_train_processed.shape[1]}")
