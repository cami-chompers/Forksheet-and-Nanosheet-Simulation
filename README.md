Forksheet and Nanosheet Transistor Simulation

Welcome to the Forksheet and Nanosheet Simulation repository. This project provides tools to simulate and analyze nanosheet and forksheet transistor designs using LTSpice and Python.

📁 Repository Contents

simulation.py – Python script for processing LTSpice simulation data, generating graphs, calculating transistor performance, and training machine learning models.

Simulation_with_epoch.py – Variant of the main script that includes performance tracking across training epochs.

nanosheet_circuit.asc – LTSpice schematic for nanosheet transistor.

forksheet_circuit.asc – LTSpice schematic for forksheet transistor.

data_set/ – Folder containing:

nanosheet_circuit.txt

forksheet_circuit.txt

⚙️ Prerequisites

1. Python

Download: python.org

Add Python to your system PATH

Verify: python --version or python3 --version

2. LTSpice

Download: LTSpice Download Page

Install and verify LTSpice runs successfully

3. Required Python Libraries

pip install pandas numpy matplotlib scikit-learn

LTSpice Simulation Setup

Open LTSpice > File > Open > Select nanosheet_circuit.asc

Run the simulation (Alt + R or "running man" icon)

Use the probe tool to select the output node and generate a waveform

Export waveform data:

Right-click waveform window > Export Data > Save as .txt

Repeat for forksheet_circuit.asc

Python Analysis Setup

Open simulation.py in your Python environment or IDE

Update test_files to point to your .txt files:

Use \ for Windows, / for macOS/Linux

Make sure the column headers in expected_columns match the headers in your .txt files (e.g., time, V(output))

Set your output folders inside main():

plot_save_path – Where images will be saved

output_path – Where results and CSVs will go

Run the script (F5 in IDLE or your editor’s run command)

Outputs

From simulation.py

Creates Graphs and Tables

Generates voltage vs. time plots and sample data tables

Combines 28 per image and saves to folder

Calculates Transistor Performance

Measures rise/fall times, propagation delay, voltage range

Saves results as .csv files

Trains Machine Learning Model

Uses Random Forest to predict behavior

Exports metrics like MAE, RMSE, R² to .csv

From Simulation_with_epoch.py

Same process as simulation.py, but trains the model in 30 steps (epochs)

Shows 4 different metric graphs for early and late training stages

⚠️ You must close each graph window manually to proceed to the next

 Notes

Double-check column names and paths if errors occur

Make sure LTSpice-exported .txt files are clean and formatted properly

Consider adding graphs of nanosheet vs forksheet comparisons and transistor evolution timelines for visual insight

Happy simulating!

