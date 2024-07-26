import serial
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
import joblib
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.exceptions

# Function to train the model
def train_model():
    # Replace this with your actual training data loading and processing
    X = np.random.rand(1000, 129)  # Dummy data for demonstration
    y = np.random.choice(['normal', 'caution', 'abnormal'], 1000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    print("Model trained successfully")
    
    # Save the trained model
    joblib.dump(model, '/home/riyap/new_myenv/modelknn.pkl')

# Load the model or train a new one if not loaded
try:
    model = joblib.load('/home/riyap/new_myenv/modelknn.pkl')
    print("Loaded pre-trained model")
except FileNotFoundError:
    print("No pre-trained model found. Training a new model.")
    train_model()
except Exception as e:
    print(f"Error loading model: {e}")

class SensorGUI:
    def __init__(self, master):
        self.master = master
        master.title("EMG and GSR Sensor Data with Seizure Prediction")
        
        # Data storage
        self.buffer_size = 500
        self.raw_emg_buffer = np.zeros(self.buffer_size)
        self.raw_gsr_buffer = np.zeros(self.buffer_size)
        self.psd_emg_buffer = np.zeros(129)  # Typical length for PSD with nperseg=256
        
        # Set up serial connection to Arduino
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200)
            print("Serial connection established")
        except Exception as e:
            print(f"Error opening serial port: {e}")
            self.ser = None
        
        # Create main frame
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left frame for plots
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right frame for predictions and traffic light
        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.line1, = self.ax1.plot([], [], 'r-')
        self.line2, = self.ax2.plot([], [], 'b-')
        
        # Set up the plots
        self.ax1.set_title('EMG Sensor Data')
        self.ax1.set_xlabel('Samples')
        self.ax1.set_ylabel('EMG Value')
        self.ax2.set_title('GSR Sensor Data')
        self.ax2.set_xlabel('Samples')
        self.ax2.set_ylabel('GSR Value')
        
        # Embed the matplotlib figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Prediction box
        self.prediction_var = tk.StringVar()
        self.prediction_var.set("No prediction yet")
        self.prediction_box = ttk.Label(self.info_frame, textvariable=self.prediction_var,
                                        background='gray', foreground='white',
                                        font=('Arial', 14, 'bold'), padding=10)
        self.prediction_box.pack(fill=tk.X, pady=10)
        
        # Traffic light
        self.traffic_light = ttk.Frame(self.info_frame)
        self.traffic_light.pack(pady=20)
        self.red_light = ttk.Label(self.traffic_light, background='gray', width=3)
        self.yellow_light = ttk.Label(self.traffic_light, background='gray', width=3)
        self.green_light = ttk.Label(self.traffic_light, background='gray', width=3)
        self.red_light.pack(pady=5)
        self.yellow_light.pack(pady=5)
        self.green_light.pack(pady=5)
        
        # Start button
        self.start_button = ttk.Button(self.info_frame, text="Start", command=self.start_streaming)
        self.start_button.pack(pady=10)
        
        # Stop button
        self.stop_button = ttk.Button(self.info_frame, text="Stop", command=self.stop_streaming)
        self.stop_button.pack(pady=10)
        
        self.is_streaming = False

    def start_streaming(self):
        self.is_streaming = True
        threading.Thread(target=self.stream_data, daemon=True).start()
        threading.Thread(target=self.update_plot, daemon=True).start()
        threading.Thread(target=self.run_predictions, daemon=True).start()

    def stop_streaming(self):
        self.is_streaming = False

    def stream_data(self):
        while self.is_streaming:
            if self.ser and self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8').strip()
                try:
                    emg, gsr = map(float, line.split(','))
                    print(f"Raw data: EMG={emg}, GSR={gsr}")  # Debug print
                    
                    # Update raw buffers
                    self.raw_emg_buffer = np.roll(self.raw_emg_buffer, -1)
                    self.raw_emg_buffer[-1] = emg
                    self.raw_gsr_buffer = np.roll(self.raw_gsr_buffer, -1)
                    self.raw_gsr_buffer[-1] = gsr
                    
                    # Calculate PSD for EMG (for prediction purposes)
                    self.psd_emg_buffer = self.process_emg_data(self.raw_emg_buffer)
                    
                except ValueError as e:
                    print(f"Error parsing data: {e}")
                    print(f"Raw line: {line}")
                
            time.sleep(0.001)  # Small delay to prevent CPU overuse

    def process_emg_data(self, data):
        # Calculate PSD
        f, psd = signal.welch(data, fs=500, nperseg=256)
        print(f"PSD shape: {psd.shape}, min: {psd.min()}, max: {psd.max()}")  # Debug print
        return psd  # Return the PSD values directly

    def run_predictions(self):
        while self.is_streaming:
            # Run prediction every 0.5 seconds
            prediction = self.predict_activity()
            self.update_gui(prediction)
            time.sleep(0.5)

    def predict_activity(self):
        features = self.psd_emg_buffer
        
        print(f"Feature shape: {features.shape}")
        print(f"Feature min: {features.min()}, max: {features.max()}")
        
        if len(features) != 129:
            print(f"Unexpected feature length: {len(features)}")
            return f"Error: Unexpected feature length ({len(features)})"
        
        features = features.reshape(1, -1)
        
        try:
            if model:
                prediction = model.predict(features)[0]
                print(f"Prediction: {prediction}")  # Debug print
                return prediction
            else:
                print("Error: Model is not loaded.")
                return "Error: Model not loaded"
        except sklearn.exceptions.NotFittedError:
            print("Error: The KNeighborsClassifier is not fitted. Please train the model.")
            return "Error: Model not trained"
        except Exception as e:
            print(f"Prediction error: {e}")
            return f"Error: {str(e)}"

    def update_gui(self, prediction):
        if prediction.startswith("Error:"):
            self.prediction_var.set(prediction)
            self.prediction_box.configure(background='gray')
        elif prediction == 'abnormal':
            self.prediction_var.set("WARNING: Abnormal activity detected!")
            self.prediction_box.configure(background='red')
            self.master.configure(background='red')
            self.update_traffic_light('red')
        elif prediction == 'caution':
            self.prediction_var.set("Caution: Elevated activity")
            self.prediction_box.configure(background='yellow')
            self.master.configure(background='yellow')
            self.update_traffic_light('yellow')
        else:
            self.prediction_var.set("No abnormal activity detected")
            self.prediction_box.configure(background='green')
            self.master.configure(background='white')
            self.update_traffic_light('green')

    def update_traffic_light(self, color):
        self.red_light.configure(background='gray')
        self.yellow_light.configure(background='gray')
        self.green_light.configure(background='gray')
        if color == 'red':
            self.red_light.configure(background='red')
        elif color == 'yellow':
            self.yellow_light.configure(background='yellow')
        else:
            self.green_light.configure(background='green')

    def update_plot(self):
        while self.is_streaming:
            # Update EMG plot (raw values)
            self.line1.set_data(range(len(self.raw_emg_buffer)), self.raw_emg_buffer)
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # Update GSR plot (raw values)
            self.line2.set_data(range(len(self.raw_gsr_buffer)), self.raw_gsr_buffer)
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            self.canvas.draw()
            self.canvas.flush_events()
            
            time.sleep(0.01)  # Update plot every 10 ms

root = tk.Tk()
gui = SensorGUI(root)
root.mainloop()
