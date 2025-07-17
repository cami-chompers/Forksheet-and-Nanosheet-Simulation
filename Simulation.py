import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 15,           # Increase default font size
    'font.weight': 'bold',     # Make fonts bold by default
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.titlesize': 15
})
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec
import time
import cProfile
import pstats


# Corrected file paths and column names
expected_columns = {
    "forksheet_circuit": ["time", "V(output)"],
    "nanosheet_circuit": ["time", "V(output)"]  # Ensure the name is consistent with the path
}

# Test files with the correct path
test_files = [
    "C:\\Users\\19562\\Downloads\\nanosheet_circuit.txt",   
    "C:\\Users\\19562\\Downloads\\forksheet_circuit.txt"    
]


# Helper function to replace units
def replace_units(value):
    return value.replace('n', 'e-9').replace('µ', 'e-6')

# Function to load and process each file
def load_and_process_data(file_path, expected_columns):
    file_name = os.path.basename(file_path).split('.')[0]
    
    # Check if the file corresponds to the expected columns
    if file_name not in expected_columns:
        raise ValueError(f"Unexpected file: {file_name}. No matching columns found.")
    
    # Initialize lists to store combined step data
    all_data = []

    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()
        current_step = None
        current_L = None
        current_W = None

        for line in lines:
            # Check for "Step Information" and extract L and W
            if line.startswith("Step Information:"):
                step_info = line
                current_L = float(replace_units(step_info.split('Lval=')[1].split()[0]))
                current_W = float(replace_units(step_info.split('Wval=')[1].split()[0]))
                current_step = True
                continue

            # If we're reading step data, append it
            if current_step and not line.startswith("Step Information"):
                values = line.strip().split()
                if len(values) == len(expected_columns[file_name]):  # Ensure correct column count
                    all_data.append(values + [current_L, current_W])  # Append L and W
                else:
                    current_step = False  # Stop if the step block ends

    columns = expected_columns[file_name] + ['L', 'W']
    data = pd.DataFrame(all_data, columns=columns).astype(float)
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

def plot_fixed_param_grouped(data, fixed_param, param_name, other_param_name, title_prefix, save_path=None, transistor_name=None):
    if transistor_name:
        data = data[data["Transistor"] == transistor_name]
        
    unique_values = sorted(data[fixed_param].unique())
    batch_size = 28  # Number of graphs per combined figure
    window_count = 0  # Counter for output image files


    for batch_start in range(0, len(unique_values), batch_size):
        batch_values = unique_values[batch_start:batch_start + batch_size]

        # Create one big figure per batch
        fig = plt.figure(figsize=(12, len(batch_values) * 8))  # More vertical space per plot-table pair
        outer_grid = gridspec.GridSpec(len(batch_values) * 2, 1)  # 2 rows per value: one for plot, one for table

        for i, value in enumerate(batch_values):
            subset = data[data[fixed_param] == value]

            # Plot
            ax_plot = plt.Subplot(fig, outer_grid[i * 2])
            ax_plot.plot(subset['time'], subset['V(output)'], label=f"{fixed_param}={value}")
            ax_plot.set_ylabel("V(output)")
            ax_plot.set_xlabel("Time (s)")
            ax_plot.set_title(f"{title_prefix} | {fixed_param}={value}")
            ax_plot.grid()
            ax_plot.legend()
            fig.add_subplot(ax_plot)

            # Table
            ax_table = plt.Subplot(fig, outer_grid[i * 2 + 1])
            ax_table.axis("off")

            # Select up to 10 rows for the table
            num_rows = 10
            if len(subset) > num_rows:
                indices = np.linspace(0, len(subset) - 1, num_rows, dtype=int)
                raw_table_data = subset.iloc[indices][['time', 'V(output)']]
            else:
                raw_table_data = subset[['time', 'V(output)']]

            formatted_data = raw_table_data.apply(lambda col: col.map(lambda x: f"{x:.3g}"))
            table = ax_table.table(
                cellText=formatted_data.values,
                colLabels=formatted_data.columns,
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.4)
            for key, cell in table.get_celld().items():
                if key[0] == 0:
                    cell.set_text_props(weight='bold')

            fig.add_subplot(ax_table)

        # Save the combined figure
        os.makedirs(save_path, exist_ok=True)
        window_count += 1
        batch_name = title_prefix.replace(" | ", "_").replace(":", "").replace(" ", "_")
        fig_filename = f"{batch_name}.png"
        fig_path = os.path.join(save_path, fig_filename)
        fig.tight_layout(pad=4.0, h_pad=3.0)  # Adds padding between rows
        fig.savefig(fig_path)
        print(f"Saved combined batch image: {fig_path}")
        plt.close(fig)

def train_model(data, target_columns, transistor_name, output_path):
    # Ensure numeric data
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    print(f"Training models for {transistor_name}, total data size: {len(data)}")

    # Get unique L and W combinations and sort by W, then L
    unique_combinations = data[['L', 'W']].drop_duplicates().sort_values(by=['L', 'W'])

    results = []

    for _, combo in unique_combinations.iterrows():
        L, W = combo['L'], combo['W']

        # Filter data for the current combination
        combo_data = data[(data['L'] == L) & (data['W'] == W)]

        if combo_data.empty:
            print(f"No data for L={L}, W={W}. Skipping.")
            continue

        # Prepare features and targets
        X = combo_data.drop(columns=target_columns)
        y = combo_data[target_columns].values.ravel()

        # Normalize features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate performance
        y_pred = model.predict(X_test)

        for i, col in enumerate(target_columns):
            # Extract the corresponding column from y_test
            y_test_col = y_test[:, i] if len(target_columns) > 1 else y_test

            # Metrics
            mae = mean_absolute_error(y_test_col, y_pred[:, i] if len(target_columns) > 1 else y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_col, y_pred[:, i] if len(target_columns) > 1 else y_pred))
            r2 = r2_score(y_test_col, y_pred[:, i] if len(target_columns) > 1 else y_pred)

            # Metrics for different conditions (on_mask, off_mask)
            threshold = 2.5
            on_mask = y_test_col >= threshold
            off_mask = y_test_col < threshold
            mae_on = mean_absolute_error(y_test_col[on_mask], y_pred[on_mask])
            mae_off = mean_absolute_error(y_test_col[off_mask], y_pred[off_mask])

            # Append results
            results.append({
                'Transistor': transistor_name,
                'Length (L)': L,
                'Width (W)': W,
                'Target': col,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAE_On': mae_on,
                'MAE_Off': mae_off
            })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Ensure correct column ordering
    results_df = results_df[['Transistor', 'Length (L)', 'Width (W)', 'Target', 'MAE', 'RMSE', 'R2', 'MAE_On', 'MAE_Off']]
    
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)    # Show all rows
    
    print(f"\n{transistor_name} model Evaluation Metrics:\n", results_df)

    # Export to CSV
    os.makedirs(output_path, exist_ok=True)
    results_df.to_csv(os.path.join(output_path, f"{transistor_name} model_evaluation.csv"), index=False)
    print(f"\n{transistor_name} model evaluation exported to {output_path}")

    # Calculate average metrics
    average_metrics = results_df[['MAE', 'RMSE', 'R2', 'MAE_On', 'MAE_Off']].mean().reset_index()
    average_metrics.columns = ['Metric', 'Average Value']

    # Display and export average metrics
    print(f"\nAverage Metrics for {transistor_name}:\n", average_metrics)
    
def find_nearest_index(data, value):
    """Find the index of the nearest value in the data array."""
    return np.abs(data - value).argmin()

def summarize_and_compare(data, output_path, transistor_name):
    # Ensure numeric data
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    # Add transistor name as a new column
    data['Transistor'] = transistor_name

    # Ensure that 'Transistor' column is in the data
    if 'Transistor' not in data.columns:
        raise ValueError("The 'Transistor' column is missing. Cannot proceed without it.")

    print(f"Summarizing data for {transistor_name}, total data size: {len(data)}")

    # Per L-W combination summary
    summary = data.groupby(['L', 'W']).agg({
        'time': ['min', 'max'],
        'V(output)': ['mean', 'std', 'min', 'max']
    }).reset_index()

    # Add the 'Transistor' column to the summary
    summary['Transistor'] = transistor_name

    # Update column names based on the aggregation
    summary.columns = [
        'Length (L)', 'Width (W)', 
        'Start Time', 'End Time', 'Mean Voltage', 
        'Voltage Std Dev', 'Min Voltage', 'Max Voltage',
        'Transistor'
    ]

    # Reorder columns to place 'Transistor' at the front
    summary = summary[['Transistor', 'Length (L)', 'Width (W)', 
                       'Start Time', 'End Time', 'Mean Voltage', 
                       'Voltage Std Dev', 'Min Voltage', 'Max Voltage']]

    # Compute average metrics across all L-W combinations for each transistor type
    avg_summary = summary.groupby('Transistor').mean(numeric_only=True).reset_index()
    avg_summary = avg_summary.rename(columns={
        'Start Time': 'Avg Start Time',
        'End Time': 'Avg End Time',
        'Mean Voltage': 'Avg Mean Voltage',
        'Voltage Std Dev': 'Avg Voltage Std Dev',
        'Min Voltage': 'Avg Min Voltage',
        'Max Voltage': 'Avg Max Voltage'
    }).drop(['Length (L)', 'Width (W)'], axis=1)

    # Performance Metrics (rise/fall times and propagation delays)
    performance_metrics = []
    for (name, L, W), subset in data.groupby(['Transistor', 'L', 'W']):
        # Ensure subset is sorted by time
        subset = subset.sort_values(by='time')

        # Extract arrays for easier manipulation
        time_array = subset['time'].values
        voltage_array = subset['V(output)'].values

        # Calculate min and max voltage
        v_min, v_max = np.min(voltage_array), np.max(voltage_array)

        # RISE TIME: Time to go from 10% to 90% of max voltage
        rise_start_voltage = v_min + 0.1 * (v_max - v_min)
        rise_end_voltage = v_min + 0.9 * (v_max - v_min)

        rise_start_idx = find_nearest_index(voltage_array, rise_start_voltage)
        rise_end_idx = find_nearest_index(voltage_array, rise_end_voltage)

        rise_start_time = time_array[rise_start_idx]
        rise_end_time = time_array[rise_end_idx]
        rise_time = max(0, rise_end_time - rise_start_time)  # Ensure non-negative

        # FALL TIME: Time to go from 90% to 10% of max voltage
        fall_start_voltage = v_max - 0.1 * (v_max - v_min)
        fall_end_voltage = v_max - 0.9 * (v_max - v_min)

        fall_start_idx = find_nearest_index(voltage_array, fall_start_voltage)
        fall_end_idx = find_nearest_index(voltage_array, fall_end_voltage)

        fall_start_time = time_array[fall_start_idx]
        fall_end_time = time_array[fall_end_idx]
        fall_time = max(0, fall_start_time - fall_end_time)  # Ensure non-negative

        # PROPAGATION DELAY: Average of 50% rise and fall times
        prop_voltage = v_min + 0.5 * (v_max - v_min)
        prop_rise_idx = find_nearest_index(voltage_array, prop_voltage)
        prop_fall_idx = find_nearest_index(voltage_array, prop_voltage)

        prop_delay_rise_time = time_array[prop_rise_idx]
        prop_delay_fall_time = time_array[prop_fall_idx]
        propagation_delay = (prop_delay_rise_time + prop_delay_fall_time) / 2

        performance_metrics.append({
            'Transistor': transistor_name,
            'Length (L)': L,
            'Width (W)': W,
            'Rise Time': rise_time,
            'Fall Time': fall_time,
            'Propagation Delay': propagation_delay
        })
        
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)    # Show all rows 

    performance_metrics_df = pd.DataFrame(performance_metrics)

    # Compute average metrics across all L-W combinations for each transistor type
    avg_metrics = performance_metrics_df.groupby('Transistor').mean(numeric_only=True).reset_index()
    avg_metrics = avg_metrics.rename(columns={
        'Rise Time': 'Avg Rise Time',
        'Fall Time': 'Avg Fall Time',
        'Propagation Delay': 'Avg Propagation Delay'
    }).drop(['Length (L)', 'Width (W)'], axis=1)

    # Print results
    print("\nPer L-W Combination Summary:\n", summary)
    print("\nAverage Summary:\n", avg_summary)
    print("\nPerformance Metrics:\n", performance_metrics_df)
    print("\nAverage Performance Metrics:\n", avg_metrics)

    # Save results to output path
    os.makedirs(output_path, exist_ok=True)
    summary.to_csv(os.path.join(output_path, f"{transistor_name} summary.csv"), index=False)
    performance_metrics_df.to_csv(os.path.join(output_path, f"{transistor_name} performance_metrics.csv"), index=False)

    print("Results exported successfully to:", output_path)


def compare_and_plot(data_ns, data_fs):
    # Add transistor type for clarity if not already present
    data_ns['Transistor'] = 'Nanosheet'
    data_fs['Transistor'] = 'Forksheet'

    combined_data = pd.concat([data_ns, data_fs], ignore_index=True)

    # Overlay plot
    plt.figure(figsize=(10, 6))
    for name, subset in combined_data.groupby('Transistor'):
        plt.plot(subset['time'], subset['V(output)'], label=name, alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("V(output)")
    plt.title("Combined Switching Speeds: Nanosheet vs Forksheet")
    plt.legend()
    plt.grid()
    plt.show()

    
# Function to export results to CSV
def export_file(data, test_name,output_path):
    data.to_csv(os.path.join(output_path, f"{test_name}_data.csv"))
    print(f"{test_name} data exported to {output_path}")

def main():
        plot_save_path = "C:\\Users\\19562\\Downloads\\stats_summary_data\\plots"

        # Load and process data for NS and FS transistors
        print("Loading and processing nanosheet data...")
        data_ns = load_and_process_data(test_files[0], expected_columns)
        print("Loading and processing forksheet data...")
        data_fs = load_and_process_data(test_files[1], expected_columns)

        # Clean and normalize data for both datasets
        print("Cleaning and normalizing nanosheet data...")
        data_ns_cleaned = clean_and_normalize(data_ns)
        print("Cleaning and normalizing forksheet data...")
        data_fs_cleaned = clean_and_normalize(data_fs)

        # Extract unique lengths (L) and widths (W) dynamically for both datasets
        print("Extracting unique lengths and widths for nanosheet...")
        unique_lengths_ns = sorted(data_ns_cleaned['L'].unique())
        unique_widths_ns = sorted(data_ns_cleaned['W'].unique())

        print("Extracting unique lengths and widths for forksheet...")
        unique_lengths_fs = sorted(data_fs_cleaned['L'].unique())
        unique_widths_fs = sorted(data_fs_cleaned['W'].unique())
        

        # Plot for each unique length with varying widths for NS and FS
        print("Generating fixed length plots for nanosheet...")
        for fixed_length in unique_lengths_ns:
            subset = data_ns_cleaned[data_ns_cleaned['L'] == fixed_length]
            title_prefix = f"Nanosheet | Fixed Length: {fixed_length:.2e}"
            save_path = os.path.join(plot_save_path, "Nanosheet_Fixed_Length")
            plot_fixed_param_grouped(subset, 'W', 'L', 'W', title_prefix, save_path=save_path)

        print("Generating fixed length plots for forksheet...")
        for fixed_length in unique_lengths_fs:
            subset = data_fs_cleaned[data_fs_cleaned['L'] == fixed_length]
            title_prefix = f"Forksheet | Fixed Length: {fixed_length:.2e}"
            save_path = os.path.join(plot_save_path, "Forksheet_Fixed_Length")
            plot_fixed_param_grouped(subset, 'W', 'L', 'W', title_prefix, save_path=save_path)

        # Plot for each unique width with varying lengths for NS and FS
        print("Generating fixed width plots for nanosheet...")
        for fixed_width in unique_widths_ns:
            subset = data_ns_cleaned[data_ns_cleaned['W'] == fixed_width]
            title_prefix = f"Nanosheet | Fixed Width: {fixed_width:.2e}"
            save_path = os.path.join(plot_save_path, "Nanosheet_Fixed_Width")
            plot_fixed_param_grouped(subset, 'L', 'W', 'L', title_prefix, save_path=save_path)

        print("Generating fixed width plots for forksheet...")
        for fixed_width in unique_widths_fs:
            subset = data_fs_cleaned[data_fs_cleaned['W'] == fixed_width]
            title_prefix = f"Forksheet | Fixed Width: {fixed_width:.2e}"
            save_path = os.path.join(plot_save_path, "Forksheet_Fixed_Width")
            plot_fixed_param_grouped(subset, 'L', 'W', 'L', title_prefix, save_path=save_path)

        

        # Define target columns for prediction
        target_columns = ['V(output)']

        # Define output directory
        output_path = "C:\\Users\\19562\\Downloads\\stats_summary_data"
        
        # Train models for NS and FS datasets
        print("\nTraining models for nanosheet data...")
        train_model(data_ns, target_columns,"Nanosheet",output_path)
        print("Training models for forksheet data...")
        train_model(data_fs, target_columns,"Forksheet",output_path)

        # Perform summary and comparison
        print("Summarizing and exporting results...")
        summarize_and_compare(data_ns,output_path,"Nanosheet")
        summarize_and_compare(data_fs,output_path,"Forksheet")
        compare_and_plot(data_ns,data_fs)


if __name__ == "__main__":

    start_time = time.time()

    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

    stats_filename = "profiling_results.prof"
    profiler.dump_stats(stats_filename)

    with open("profiling_summary.txt", "w") as f:
        stats = pstats.Stats(stats_filename, stream=f)
        stats.strip_dirs().sort_stats("cumulative").print_stats(20)

    print("\nProfiling complete! Results saved to 'profiling_results.prof' and 'profiling_summary.txt'.")
