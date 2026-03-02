# Place to store global parameters
from dataclasses import dataclass

@dataclass
class Parameters:
    # Initial position
    X0: float
    Y0: float
    # Location of missing person
    XL: float
    YL: float

    R_SCAN: float # Radius of the "scanner"
    T: float # Flight time from any stopped position to next target

    M: float # Drone mass
    EH: float # Hovering energy per second
    ES: float # Constant energy to perform one scan
    E_MAX: float # Max energy budget

    DT: float
    EPS: float # Threshold used to determine if the drone should return midway (if energy margin < eps)

    # Time to come to a stop when turning midway
    @property
    def TS(self):
        return self.T / 20

    # Time to return back to the charging station
    @property
    def TR(self):
        return self.T * 5