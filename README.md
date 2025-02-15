# Neuromuscular Disorder Detection System

## Technical Details
- **Model Architecture**: KNN Classifier with 75% accuracy, compared against Random Forest (92%), Logistic Regression (42%), and SVM (62%)
- **Signal Processing**: 
  - Digital bandpass filter (75-145 Hz)
  - Power Spectral Density (PSD) analysis
  - Continuous Wavelet Transform (CWT)
- **Data Streaming**: Lab Streaming Layer (LSL) at 500 Hz sampling rate
- **Real-time Processing**: Continuous prediction every second using 500 EMG data points

## System Architecture
1. **Data Acquisition Layer**:
   - Arduino Uno with EMG and GSR sensors
   - Analog to Digital Conversion
   - Real-time data streaming at 500 Hz

2. **Processing Layer** (Raspberry Pi 4B):
   - Flask server for web interface
   - SocketIO for real-time data broadcasting
   - Pre-trained ML model for classification
   - Digital bandpass filtering
   
3. **Interface Layer**:
   - Local GUI (Tkinter + Matplotlib)
   - Web interface (Flask + SocketIO)
   - Real-time visualization
   - Status indication system

## Repository Structure

- `/GUI.py`: Contains the code for the Raspberry Pi GUI and data processing.
- `/ml_model`: Includes the machine learning model and related scripts.
- `/GSR_test.ino`: Arduino code for sensor data acquisition.

## Features

- Real-time data visualization of EMG and GSR signals.
- Live predictions for neuromuscular disorders using EMG data.
- Remote accessibility of the GUI for monitoring.

## Setup and Installation

(Include steps for setting up the project, including hardware requirements and software installation instructions.)

## Usage

(Provide instructions on how to run the project, including any command-line instructions or GUI operations.)

## Remote Access

To access the GUI remotely:
1. Ensure your device is on the same network as the Raspberry Pi.
2. (Include steps to connect to the Raspberry Pi's server)

## Contributors

(List the names of project contributors)

## Acknowledgments

- Thanks to [Medical Institution Name] for providing the dataset used in training the ML model.

## Future Work

- Incorporate GSR data into the ML model for more comprehensive analysis.
- Expand the model to include data from additional muscle groups and body parts.

## License

(Include appropriate license information)
#EMG #GSR #MachineLearning #RaspberryPi #Arduino #HealthTech
