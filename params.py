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

    M: float # Drone mass
    EH: float # Hovering energy per second
    ES: float # Constant energy to perform one scan
    E_MAX: float # Max energy budget

    DT: float
    EPS: float # Threshold used to determine if the drone should return midway (if energy margin < eps)

    SOLVE_IVP_COUNTER: int
    R_MAX: float # Maximum radius to energy circumference, determined from first linear search

    # Flight time from any stopped position to next target
    @property
    def T(self):
        return (9 * self.M / (2 * self.EH)) ** (1/3) * self.R_SCAN ** (2/3)

    # Time to come to a stop when turning midway
    @property
    def TS(self):
        return self.T / 20