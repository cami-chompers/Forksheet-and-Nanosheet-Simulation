# Optimization of Forksheet and Nanosheet Transistors Through Parametric Simulations and Machine Learning
---

## Description
This project presents a combined approach using LTSpice simulations and a Python-based machine learning (ML) framework to analyze and optimize nanosheet (NS) and forksheet (FS) transistor architectures.  
The code supports:
- Device-level simulations
- Automated extraction of key transistor metrics from waveform data
- ML model training for rapid performance prediction

Focus parameters include rise time, fall time, propagation delay, and voltage stability across different transistor geometries.

---
## Dataset Information
- The dataset consists of simulation output waveforms and extracted electrical switching speed performance metrics (rise/fall time, delay, voltage stability) for NS and FS transistor designs over a range of lengths and widths.
- Data is stored in CSV files with features corresponding to transistor dimensions and labels including measured performance metrics.

---

## Code Information
The repository includes:

- **simulation.py**: A Python script for processing and analyzing LTSpice simulation data, generating visualizations, and evaluating transistor performance metrics.  
- **nanosheet_circuit.asc**: LTSpice schematic for nanosheet transistor simulations.  
- **forksheet_circuit.asc**: LTSpice schematic for forksheet transistor simulations.  
- **Simulation_with_epoch.py**: Python script similar to `simulation.py` with additional model training checkpoints and performance visualizations over epochs.  
- **data_set folder**: Contains `nanosheet_circuit.txt` and `forksheet_circuit.txt` — LTSpice simulation data files for each transistor used as input for the Python scripts.
- **data_summary folder**: Excel documents for all of the outputs for each transistor. Minimums, maximums, and averages are calculated at the bottom of the Excel document.

Modular code design allows easy adaptation to additional transistor configurations or metrics.

---

## Prerequisites

Before using this repository, ensure the following are installed on your system and are aware of the size necessities:

1.**Size Requirements** 
- **Python IDLE (Python 3.13) Software**: ~153 MB
- **Python Launcher Software**: ~1.48 MB
- **Python libraries compined (pandas numpy matplotlib scikit-learn)** ~98.2 MB
- **LTSpice Software (24.0.12.0)**: ~152 MB
- **simulation.py**: ~20.7 KB
- **nanosheet_circuit.asc**: ~1.30 KB
- **forksheet_circuit.asc**: ~1.33 KB
- **Simulation_with_epoch.py**: ~26.3 KB
- **Data extracted from the nanosheet LTSpice circuit**: ~16.3 MB
-  **Data extracted from the forksheet LTSpice circuit**: ~17.8 MB
- **Output folder that includes all the data figures and evaluations from simulation.py**: ~221 MB
- **Output folder that includes all the data figures and evaluations from Simulation_with_epoch.py**: ~221 MB

   Total size requirements: 881.33 MB

2. **Python**  
   - Download from [python.org](https://www.python.org/downloads/)  
   - Follow installation instructions, making sure to add Python to your system’s PATH  
   - Verify installation by running:  
     ```bash
     python --version
     ```
3. **LTSpice**  
   - Download from [LTSpice Download Page](https://ez.analog.com/design-tools-and-calculators/ltspice/w/faqs-docs/32232/ltspice-24-download-and-release-notes)  
   - Follow installation instructions and verify by launching LTSpice.

4. **Required Python Libraries**  
   Install the necessary libraries using:  
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
   

## Step 1: LTSpice Simulations

- Open LTSpice and navigate to **File > Open**.  
- Load the circuit file `nanosheet_circuit.asc`.  
- Click the **Run** button (running man icon) or press `Alt + R` to simulate.  
- Use the probe tool to click on the output node and visualize waveforms⚠️ **Important:** Epilepsy warning as the waveform switches color at a fast speed.  
- After the waveform stops falsing colors or it's done processing, export the simulation data:  
  - Right-click on the waveform window → **Export Data** → save as `.txt`.  
- Repeat the above steps for `forksheet_circuit.asc`.  

## Step 2: Python Analysis

- Open `simulation.py` in your Python IDE (e.g., IDLE).  
- Update the file paths in the `test_files` list to point to your own `.txt` simulation data files exported from LTSpice.  
  - Use double backslashes (`\\`) on Windows or forward slashes (`/`) on macOS/Linux.  
- Check the `expected_columns` section to ensure your file names and column headers (e.g., `["time", "V(output)"]`) match your `.txt` files.  
- Set output folder paths in the global variable of stats_summary_folder, but don't change the "\\stats_summary_data" of the variable as it will be the name of the folder which stores the outputs:    
- Run the script:  
  - In IDLE, navigate to **Run > Run Module** or press `F5`.  

## Outputs

The script will:

- **Create graphs and tables**  
  It generates voltage vs. time plots and sample data tables grouped into images (28 plots per image), saving them to your specified folder.  

- **Calculate and save transistor performance metrics**  
  Measures rise time, fall time, propagation delay, and minimum/maximum voltages, saving results as CSV files.  

- **Train a machine learning model**  
  Uses a Random Forest model to predict voltage behavior, evaluates using MAE, RMSE, R², and exports evaluation results as CSV files.  

## About Simulation_with_epoch.py

This version trains the model over 30 epochs and shows 4 performance graphs.

⚠️ **Important:** You need to close each graph window one by one to continue to the next. There will be 8 performance graphs total, 4 for each transistor.

## Notes

- If you encounter errors about missing columns or incorrect file paths, make sure your LTSpice-exported `.txt` files are correctly formatted and update the paths in `simulation.py`.  
