import numpy as np

# TODO: I want to integrate this to make handling numpy arrays a little easier in the core files
# and abstract that away to class-level stuff
class LinearSearchPhaseHistory:
    def __init__(self):
        self.times = None # All timesteps
        self.trajectory = None # Full trajectory, where each entry contains state [x, y, vx, vy, e]

        self.e_used = None # Cumulative energy used at each timestep

        self.e_turn = None # Energy required to return at each turn time check
        self.e_turn_times = None # List of times where return energy was checked

         # Index of arrays where drone *decided* to turn, whether that be due to energy or finding the person
        self.turn_index = None
        # Index of arrays where drone was stopped before starting to return
        # (either after coming to a stop for energy reasons or at the last scan location where the person was found)
        self.stop_index = None
        # Indices of arrays where scans occurred
        self.scan_indices = []

        self.turned = False # Note: this should always be true by the end of the linear search phase
        self.located = False # Found the person
        self.direction = None # Direction of search
        self.angle = None # Angle to travel at