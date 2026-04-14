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
    {
        "event_id": None,
        "start": "09:00",
        "end": "10:00",
        "task": "Team Meeting",
        "description": "Weekly team sync covering priorities, blockers, and coordination.",
        "intensity": "medium",
        "movable": False,
    },
    {
        "event_id": None,
        "start": "10:00",
        "end": "12:00",
        "task": "Deep Work Session",
        "description": "Focused individual work requiring sustained concentration and minimal interruptions.",
        "intensity": "high",
        "movable": True,
    },
    {
        "event_id": None,
        "start": "12:00",
        "end": "13:00",
        "task": "Lunch Break",
        "description": "Midday meal and recovery break.",
        "intensity": "low",
        "movable": False,
    },
    {
        "event_id": None,
        "start": "13:00",
        "end": "15:00",
        "task": "Project A",
        "description": "High-priority project execution block with deliverable progress expected.",
        "intensity": "high",
        "movable": True,
    },
    {
        "event_id": None,
        "start": "15:00",
        "end": "16:00",
        "task": "Project B",
        "description": "Medium-intensity project work such as follow-ups, implementation, or coordination.",
        "intensity": "medium",
        "movable": True,
    },
    {
        "event_id": None,
        "start": "16:00",
        "end": "17:00",
        "task": "Wrap-up and Planning for Tomorrow",
        "description": "Admin wrap-up, status notes, and planning for the next day.",
        "intensity": "low",
        "movable": True,
    },
]
# This keeps the local clustering model and the LLM layer aligned on what each cluster means.
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


def create_sense():
    # sense_emu already handles the emulation role on non-Pi devices.
    # If the GUI backend is unavailable, keep the program running and let Sensor fall back to defaults.
    try:
        return SenseHat()
    except Exception as exc:
        print(f"Sense HAT initialization failed ({exc}). Sensor values will fall back to defaults.")
        return None


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
        if self.sense is not None:
            self.sense.clear()
        
    def get_temperature(self):
        if self.sense is None:
            return 20.0
        try:
            temperature = self.sense.get_temperature()
        except Exception as e:
            print(f"Error reading temperature: {e}. Using default value of 20.0°C.")
            temperature = 20.0
        return temperature
    
    def get_humidity(self):
        if self.sense is None:
            return 60.0
        try:
            humidity = self.sense.get_humidity()
        except Exception as e:
            print(f"Error reading humidity: {e}. Using default value of 60.0%.")
            humidity = 60.0
        return humidity
    
    def get_pressure(self):
        if self.sense is None:
            return 1013.25
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
        if self.sense is None:
            return
        self.sense.clear()
        self.sense.set_pixels(pixels)


class SenseHatController:
    # This controller is the bridge between the local clustering model and the later LLM stage.
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
        # Keep this as mock data during testing so different sleep scenarios can be injected easily.
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
        if cluster not in [0, 1, 2]:
            print(f"Warning: cluster {cluster} is out of expected range.")
            return ""

        cluster_profile = get_cluster_profile(cluster)
        if cluster_profile is None:
            raise ValueError(f"No cluster profile found for cluster {cluster}")
        
        if self.sense is not None:
            self.sense.clear()
        temperature = self.sensor.get_temperature()
        humidity = self.sensor.get_humidity()
        pressure = self.sensor.get_pressure()
        
        schedule = "\n".join(
            f"{item['start']} - {item['end']}: {item['task']}" for item in get_mock_schedule()
        )
        
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
                    W, W, W, W, W, W, W, W,
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
                    W, W, W, W, W, W, W, W,
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
                    W, W, W, W, W, W, W, W,
                ]
            case _:
                pixels = [W] * 64
        self.led_matrix.display_message(pixels)
        return


def main() -> int:
    controller = SenseHatController()
    sleep_data = controller.fetch_sleep_data()
    cluster = controller.predict_sleep_quality(sleep_data)
    prompt = controller.generate_prompt(cluster)

    print(f"Predicted cluster: {cluster}")
    print(prompt)
    controller.led_display(cluster)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
