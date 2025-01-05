Welcome to the Forkshheet and Nanoshet Simulation repository. This project provides tools to simulate and analyze nanosheet and forksheet transistor designs using LTSpice and Python.

Repository Contents
- simulation.py: A Python script for processing and analyzing LTSpice simulation data, generating visualizations, and evaluating transistor performance metrics.
- nanosheet_circuit.asc: LTSpice schematic for nanosheet transistor simulations.
- forksheet_circuit.asc: LTSpice schematic for forksheet transistor simulations.

Prerequisites
Before using this repository, ensure the following tools are installed on your system:
1. Python 
   - Download from the official Python website: (https://www.python.org/downloads/)
   - Follow the installation instructions, ensuring you add Python to your system’s PATH.
   - Verify installation by running `python --version` or `python3 --version` in your terminal or command prompt.

2. LTSpice
   - Download from the LTSpice download page:(https://ez.analog.com/design-tools-and-calculators/ltspice/w/faqs-docs/32232/ltspice-24-download-and-release-notes).
   - Follow the installation instructions and verify by launching LTSpice from your start menu or desktop shortcut.

3. Required Python Libraries 
   - Install the necessary libraries using the following command: pip install pandas numpy matplotlib scikit-learn tk

Instructions
Step 1: LTSpice Simulations
1. Open LTSpice and navigate to File > Open.
2. Load the circuit file `nanosheet_circuit.asc`. The nanosheet transistor schematic will appear.
3. Click the Run button (depicted as a "running man" icon) or press `Alt + R` to simulate the circuit.
4. To visualize waveforms, use the probe tool and click on the output node in the circuit.  
5. Export the simulation data:  
   - Right-click on the waveform window, select Export Data, and save it as a `.txt` file.  
6. Repeat these steps for `forksheet_circuit.asc` to simulate the forksheet transistor design.

Step 2: Python Analysis
1. Open the `simulation.py` script in your preferred Python editor (e.g., IDLE or VS Code).
2. Update file paths in the `test_files` section of the script to match the paths of the exported `.txt` files from LTSpice.  
   - For Windows: Use double backslashes (`\\`).  
   - For macOS/Linux: Use single forward slashes (`/`).
3. Update the `expected_columns` section with the appropriate column names and set the desired output path for results in the `output_path` variable in the `main()` function.
4. Run the script:  
   - In IDLE: Navigate to Run > Run Module or press `F5`.  

Outputs
The script will:
- Generate voltage vs. time plots for switching characteristics.
- Export performance summaries as `.csv` files, including metrics like rise time, fall time, and propagation delays.
- Evaluate machine learning models and display key metrics like MAE, MAE Off, MAE On, RMSE, and R-squared (R2).

Example Outputs
- Voltage vs. Time Plots: Visualize switching characteristics of nanosheet and forksheet designs.
- Performance Metrics: CSV files summarizing transistor behaviors.
- Model Evaluations: Machine learning predictions of transistor performance.

Notes
- If you encounter issues with missing columns or file paths, ensure the exported LTSpice `.txt` files are correctly formatted and paths are updated in `simulation.py`.
- Visual figures for comparing nanosheet and forksheet designs and transistor evolution timelines can be added to your workflow for enhanced analysis.
