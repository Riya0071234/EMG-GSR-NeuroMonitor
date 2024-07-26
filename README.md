# Neuromuscular Disorder Detection System

This project implements a real-time neuromuscular disorder detection system using EMG (Electromyography) and GSR (Galvanic Skin Response) sensors. The system runs on a Raspberry Pi, with data acquisition from an Arduino Uno.

## Project Overview

The system consists of several key components:

1. **Sensor Data Acquisition**: EMG and GSR data are collected using sensors connected to an Arduino Uno.

2. **Data Processing**: The sensor data is fed into a Raspberry Pi for processing and analysis.

3. **Machine Learning Model**: An ML model, trained on a dataset from a medical institution, processes the EMG data to detect potential neuromuscular disorders. The model is trained specifically for right hand muscle data.

4. **Graphical User Interface (GUI)**: A real-time GUI on the Raspberry Pi displays:
   - EMG signal plot
   - GSR signal plot
   - Prediction results from the ML model

5. **Remote Access**: The GUI can be accessed remotely from any device on the same network as the Raspberry Pi.

## Repository Structure

- `/sensor_gui`: Contains the code for the Raspberry Pi GUI and data processing.
- `/ml_model`: Includes the machine learning model and related scripts.
- `/arduino`: Arduino code for sensor data acquisition.

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
