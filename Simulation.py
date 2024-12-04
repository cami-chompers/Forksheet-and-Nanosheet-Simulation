import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Corrected file paths and column names
expected_columns = {
    "Switching Speeds_NS": ["time", "V(output)", "V(n003)", "I(Vdd)"],
    "Switching Speeds_FS_drafft": ["time", "V(output)", "V(n003)", "I(Vdd)"]  # Ensure the name is consistent with the path
}

# Test files with the correct path for "Switching Speeds_FS_drafft.txt"
test_files = [
    "C:\\Users\\19562\\Downloads\\NS\\Switching Speeds_NS.txt",  # Corrected path
    "C:\\Users\\19562\\Downloads\\FS\\Switching Speeds_FS_drafft.txt"   # Corrected path
]

# Function to load and process each file
def load_and_process_data(file_path, expected_columns):
    file_name = os.path.basename(file_path).split('.')[0]  # Extract the file name without extension
    
    # Check if the file corresponds to the expected columns
    if file_name not in expected_columns:
        raise ValueError(f"Unexpected file: {file_name}. No matching columns found.")
    
    # Load data from the file
    try:
        data = pd.read_csv(file_path, sep=r'\s+', comment="S", header=None, encoding="ISO-8859-1")
    except UnicodeDecodeError:
        print(f"Error reading file {file_path}. Trying with a different encoding...")
        data = pd.read_csv(file_path, sep=r'\s+', comment="S", header=None, encoding="ISO-8859-1")

    # Print column names for debugging
    print(f"Columns in {file_name}: {data.columns}")
    
    # Add column names
    data.columns = expected_columns[file_name]
    
    # Extract "Step Information" for L and W values
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()
        step_info = next(line for line in lines if line.startswith('Step Information'))
        L_val = float(step_info.split('Lval=')[1].split()[0].replace('n', 'e-9').replace('µ', 'e-6'))
        W_val = float(step_info.split('Wval=')[1].split()[0].replace('µ', 'e-6'))
    
    # Add L and W values as columns to the data
    data['L'] = L_val
    data['W'] = W_val
    
    return data

# Function to clean and normalize data
def clean_and_normalize(data):
    # Select numeric columns only
    numeric_data = data.select_dtypes(include=['number'])
    
    # Fill NaN values with the mean of each numeric column
    numeric_data.fillna(numeric_data.mean(), inplace=True)
    
    # Replace the original data with the cleaned numeric data
    data[numeric_data.columns] = numeric_data
    return data


# Function to visualize the data
def visualize_data(data, test_name): 
    # Check for columns relevant to the test type and plot accordingly

    if all(col in data.columns for col in ["time", "V(output)", "V(n003)", "I(Vdd)"]): # Switching Speeds
        plt.figure(figsize=(8, 6))
        # Plot V(output) vs Time
        plt.plot(data['time'], data['V(output)'], label=f"{test_name} - V(output) vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("V(output) (Volts)")
        plt.title(f"{test_name} - Switching Speed Measurement")
        
    else:
        print(f"Warning: No relevant columns found for {test_name}. Skipping plot.")
    
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model(data, target_columns):
    # Ensure numeric data
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()

    # Debugging columns and basic stats
    print(f"Columns in dataset ({target_columns}):", data.columns)
    print("Target column stats:")
    for col in target_columns:
        print(f"\n{col}:\n{data[col].describe()}")

    # Features and multiple targets
    X = data.drop(columns=target_columns)
    y = data[target_columns]  # Multiple targets as a DataFrame


    # Normalize features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Cross-validation (for single target column only; adjust for multiple if needed)
    cv_scores = []
    for col in target_columns:
        scores = cross_val_score(model, X_train, y_train[col], cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-scores).mean()
        cv_scores.append((col, cv_rmse))
        print(f"Cross-Validation RMSE for {col} (average): {cv_rmse:.4f}")

    # Prediction and evaluation for each target
    y_pred = model.predict(X_test)
    for i, col in enumerate(target_columns):
        mae = mean_absolute_error(y_test[col], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[col], y_pred[:, i]))
        print(f"\n{col}:")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

    return model, X_test, y_test, y_pred




# Function to export results to CSV
def export_results(data, test_name):
    output_path = f"C:/Users/19562/Downloads/{test_name}_results.csv"
    data.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")

def main():
    for file_path in test_files:
        # Load and process data for each file
        data = load_and_process_data(file_path, expected_columns)
        
        # Clean and normalize the data
        data_cleaned = clean_and_normalize(data)
        
        # Visualize the data
        visualize_data(data_cleaned, os.path.basename(file_path))
        file_name = os.path.basename(file_path)
        
        # Train models for each test
        if "Switching Speeds_NS" in file_name:
            target_columns = ['V(output)']
        if "Switching Speeds_FS_drafft" in file_name:
            target_columns = ['V(output)']
        
        # Now call the model training function with the appropriate target columns
        model, X_test, y_test, y_pred = train_model(data_cleaned, target_columns)
        
        # Export results to CSV
        export_results(data_cleaned, os.path.basename(file_path))

# Run the main function
if __name__ == "__main__":
    main()
