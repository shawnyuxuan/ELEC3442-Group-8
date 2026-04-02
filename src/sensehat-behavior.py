import time, datetime
from typing import Literal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os, sys
import pandas as pd
import pickle

def is_raspberry_pi():
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'Raspberry Pi' in f.read()
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"The method determines the system is not a Raspberry Pi, but due to an unexpected error: {e}")
        return False

if is_raspberry_pi():
    from sense_hat import SenseHat
else:
    try:
        from sense_emu import SenseHat
    except ImportError:
        print("sense_emu library not found. Install it using 'pip install sense_emu' to run this code on non-Raspberry Pi devices.")
        sys.exit(1)

root_path = os.path.dirname(os.path.abspath(__file__))
sense = SenseHat()

class Sensor:
    def __init__(self):
        self.sense = sense
        self.sense.clear()
        
    def get_temperature(self):
        try:
            temperature = self.sense.get_temperature()
        except Exception as e:
            print(f"Error reading temperature: {e}. Using default value of 20.0°C.")
            temperature = 20.0
        return temperature
    
    def get_humidity(self):
        try:
            humidity = self.sense.get_humidity()
        except Exception as e:
            print(f"Error reading humidity: {e}. Using default value of 60.0%.")
            humidity = 60.0
        return humidity
    
    def get_pressure(self):
        try:
            pressure = self.sense.get_pressure()
        except Exception as e:
            print(f"Error reading pressure: {e}. Using default value of 1013.25 hPa.")
            pressure = 1013.25
        return pressure

class LEDMatrix:
    def __init__(self):
        self.sense = sense

    def display_message(self, pixels):
        self.sense.clear()
        self.sense.set_pixels(pixels)

class SenseHatController:
    def __init__(self):
        self.sensor = Sensor()
        self.led_matrix = LEDMatrix()
        self.model, self.scaler = self.load_model()
    
    def load_model(self) -> tuple[KMeans, StandardScaler]:
        try:
            model, scaler = pickle.load(open(f"{root_path}/../output/local_model.pkl", "rb"))
        except FileNotFoundError:
            print("Model not found. Please run routine-learning.py to train the model first.")
            exit(1)
        return model, scaler

    def fetch_sleep_data(self) -> pd.DataFrame:
        #TODO: Implement this function either via Apple HealthKit API or by manually input.
        
        # Mock sleep data
        data = pd.DataFrame({
            "start_sin": [-0.1],
            "CORE": [300],
            "DEEP": [90],
            "REM": [80],
            "AWAKE": [10],
            "UNSPECIFIED": [0],
            "weekday": [1]
        })
        return data
    
    def predict_sleep_quality(self, sleep_data: pd.DataFrame) -> int:
        sleep_data_scaled = self.scaler.transform(sleep_data)
        cluster = self.model.predict(sleep_data_scaled)[0]
        return cluster

    def generate_prompt(self, cluster: Literal[0, 1, 2]):
        #XXX: Hardcode?
        if cluster not in [0, 1, 2]:
            print(f"Warning: cluster {cluster} is out of expected range.)")
            return ""
        
        quality_map = {
            0: "user was well-rested with balanced sleep stages. Suggest a \"Deep Work\" focused schedule.",
            1: "user slept late and for a shorter duration. Potential for \"brain fog.\" Suggest \"Administrative/Light\" tasks in the morning and moving critical tasks to the afternoon.",
            2: "user had a massive sleep debt recovery. While duration was long, the user might experience \"Sleep Inertia\" (grogginess). Suggest a slow start with caffeine/physical activity and avoiding heavy deadlines until later in the day."
        }
        
        self.sense.clear()
        temperature = self.sensor.get_temperature()
        humidity = self.sensor.get_humidity()
        pressure = self.sensor.get_pressure()
        
        #TODO: Replace this with actual schedule fetching logic, e.g., from Apple Calendar API.
        # Mock schedule
        schedule = "9:00 AM - 10:00 AM: Team Meeting\n10:00 AM - 12:00 PM: Deep Work Session\n12:00 PM - 1:00 PM: Lunch Break\n1:00 PM - 3:00 PM: Project A\n3:00 PM - 4:00 PM: Project B\n4:00 PM - 5:00 PM: Wrap-up and Planning for Tomorrow"
        
        #TODO: Provisional. Refined prompt engineering needed.
        prompt = f"""Last night, the {quality_map[cluster]}.
        Current environmental conditions are: Temperature: {temperature:.1f}°C, Humidity: {humidity:.1f}%, Pressure: {pressure:.1f} hPa.
        The original schedule today includes: {schedule}.
        Please suggest an optimized schedule for today based on the sleep quality and current environmental conditions."""
        
        return prompt
    
    def led_display(self, cluster: Literal[0, 1, 2]):
        G = (0, 255, 0)
        Y = (255, 255, 0)
        R = (255, 0, 0)
        W = (255, 255, 255)
        B = (0, 0, 0)
        match cluster:
            case 0:
                # A smile :)
                pixels = [
                    W, W, W, W, W, W, W, W,
                    W, G, G, W, W, G, G, W,
                    W, G, G, W, W, G, G, W,
                    W, W, W, W, W, W, W, W,
                    G, W, W, W, W, W, W, G,
                    W, G, G, W, W, G, G, W,
                    W, W, G, G, G, G, W, W,
                ]
            case 1:
                # A frown :(
                pixels = [
                    W, W, W, W, W, W, W, W,
                    W, Y, Y, W, W, Y, Y, W,
                    W, Y, Y, W, W, Y, Y, W,
                    W, W, W, W, W, W, W, W,
                    W, W, Y, Y, Y, Y, W, W,
                    W, Y, Y, W, W, Y, Y, W,
                    Y, W, W, W, W, W, W, Y,
                ]
            case 2:
                # A neutral face :|
                pixels = [
                    W, W, W, W, W, W, W, W,
                    W, R, R, W, W, R, R, W,
                    W, W, W, W, W, W, W, W,
                    W, W, W, W, W, W, W, W,
                    W, W, W, W, W, W, W, W,
                    W, R, R, R, R, R, R, W,
                    W, R, R, R, R, R, R, W,
                ]
            case _:
                # Default to white for unknown cluster
                pixels = [W] * 64
        self.led_matrix.display_message(pixels)
        return