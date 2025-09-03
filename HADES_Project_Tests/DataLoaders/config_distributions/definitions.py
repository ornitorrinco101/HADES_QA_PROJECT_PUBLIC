# definitions.py
from typing import Dict, Callable, Optional, List  # Added List here
import numpy as np

class Distribution:
    def __init__(
        self,
        name: str,
        data_generator: Callable[[], np.ndarray],
        hist_params: Dict,
        anomalies: Optional[List[Dict]] = None,
        second_downsampler=None
    ):
        self.name = name
        self.generate_data = data_generator
        self.hist_params = hist_params
        self.anomalies = anomalies or []
        self.second_downsampler = second_downsampler

def anomaly(name: str, func: Callable) -> Dict:
    return {"name": name, "func": func}