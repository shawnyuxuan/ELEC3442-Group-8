import argparse
import time, datetime
from typing import Literal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os, sys
import pandas as pd
import pickle


DEFAULT_MODEL_CANDIDATES = [
    "jayden_model.pkl",
    "local_model.pkl",
]
DEFAULT_MOCK_SCHEDULE = [
    {"start": "09:00", "end": "10:00", "task": "Team Meeting", "intensity": "medium", "movable": False},
    {"start": "10:00", "end": "12:00", "task": "Deep Work Session", "intensity": "high", "movable": True},
    {"start": "12:00", "end": "13:00", "task": "Lunch Break", "intensity": "low", "movable": False},
    {"start": "13:00", "end": "15:00", "task": "Project A", "intensity": "high", "movable": True},
    {"start": "15:00", "end": "16:00", "task": "Project B", "intensity": "medium", "movable": True},
    {"start": "16:00", "end": "17:00", "task": "Wrap-up and Planning for Tomorrow", "intensity": "low", "movable": True},
]
CLUSTER_PROFILES = {
    0: {
        "label": "Deep Work Ready",
        "summary": "user was well-rested with balanced sleep stages",
        "recommended_work_style": "Prioritize deep work and cognitively demanding tasks early in the day.",
        "likely_risks": [
            "overcommitting to too many hard tasks",
            "underusing a high-energy morning window",
        ],
    },
    1: {
        "label": "Brain Fog Risk",
        "summary": "user slept late and for a shorter duration",
        "recommended_work_style": "Start with administrative or light work, then move critical tasks to the afternoon.",
        "likely_risks": [
            "slower task switching in the morning",
            "reduced focus on deep work before noon",
            "higher mental fatigue during long meetings",
        ],
    },
    2: {
        "label": "Sleep Inertia Risk",
        "summary": "user had a long recovery-style sleep and may still feel groggy after waking",
        "recommended_work_style": "Use a slow start, add activation or light movement, and delay heavy deadlines until later.",
        "likely_risks": [
            "slow warm-up after waking",
            "reduced sharpness in the early morning",
            "low momentum at the start of the day",
        ],
    },
}

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


class DummySenseHat:
    def clear(self):
        return

    def get_temperature(self):
        return 20.0

    def get_humidity(self):
        return 60.0

    def get_pressure(self):
        return 1013.25

    def set_pixels(self, pixels):
        self.last_pixels = pixels


def create_sense():
    try:
        return SenseHat()
    except FileNotFoundError as exc:
        print(f"Sense HAT emulator GUI is unavailable ({exc}). Falling back to dummy sensor values.")
        return DummySenseHat()
    except Exception as exc:
        print(f"Sense HAT initialization failed ({exc}). Falling back to dummy sensor values.")
        return DummySenseHat()


sense = create_sense()


def resolve_model_path(model_path: str | None = None) -> str:
    if model_path:
        return model_path

    output_dir = os.path.join(root_path, "..", "output")
    for candidate in DEFAULT_MODEL_CANDIDATES:
        resolved = os.path.join(output_dir, candidate)
        if os.path.exists(resolved):
            return resolved

    return os.path.join(output_dir, "local_model.pkl")


def get_cluster_profile(cluster: int) -> dict[str, object] | None:
    return CLUSTER_PROFILES.get(cluster)


def get_mock_schedule() -> list[dict[str, object]]:
    return [item.copy() for item in DEFAULT_MOCK_SCHEDULE]

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
    def __init__(self, model_path: str | None = None):
        self.sense = sense
        self.sensor = Sensor()
        self.led_matrix = LEDMatrix()
        self.model_path = resolve_model_path(model_path)
        self.model, self.scaler = self.load_model()
    
    def load_model(self) -> tuple[KMeans, StandardScaler]:
        try:
            with open(self.model_path, "rb") as handle:
                model, scaler = pickle.load(handle)
        except FileNotFoundError:
            print(
                f"Model not found: {self.model_path}. "
                "Please run routine-learning.py to train the model first."
            )
            exit(1)
        print(f"Loaded model: {self.model_path}")
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

        cluster_profile = get_cluster_profile(cluster)
        assert cluster_profile is not None
        
        self.sense.clear()
        temperature = self.sensor.get_temperature()
        humidity = self.sensor.get_humidity()
        pressure = self.sensor.get_pressure()
        
        schedule = "\n".join(
            f"{item['start']} - {item['end']}: {item['task']}" for item in get_mock_schedule()
        )
        
        #TODO: Provisional. Refined prompt engineering needed.
        prompt = f"""Last night, the {cluster_profile['summary']}. {cluster_profile['recommended_work_style']}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sense HAT sleep-quality predictor.")
    parser.add_argument(
        "--model",
        help="Path to the trained model pickle. Defaults to output/jayden_model.pkl if present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controller = SenseHatController(model_path=args.model)
    sleep_data = controller.fetch_sleep_data()
    cluster = controller.predict_sleep_quality(sleep_data)
    prompt = controller.generate_prompt(cluster)

    print(f"Predicted cluster: {cluster}")
    print(prompt)
    controller.led_display(cluster)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
