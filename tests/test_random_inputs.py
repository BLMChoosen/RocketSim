
import sys
import os
import random
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add visualizer to path so its internal imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'visualizer')))

from visualizer.sim_wrapper import SimWrapper
from visualizer.main import QRSVWindow, QRSVGLWidget
import visualizer.ui as ui
from PyQt5 import QtWidgets

class SimpleControls:
    def __init__(self):
        self.throttle = 0.0
        self.steer = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.jump = False
        self.boost = False
        self.handbrake = False

class RandomSimWrapper(SimWrapper):
    def __init__(self, n_envs=1, max_cars=6):
        super().__init__(n_envs, max_cars)
        self.step_count = 0
        self.current_controls = SimpleControls()
        self.change_interval = 30 # Change inputs every 0.25s (at 120hz)

    def step(self, user_controls=None):
        self.step_count += 1
        
        if self.step_count % self.change_interval == 0:
            # Randomize controls
            self.current_controls.throttle = random.uniform(-1.0, 1.0)
            self.current_controls.steer = random.uniform(-1.0, 1.0)
            self.current_controls.pitch = random.uniform(-1.0, 1.0)
            self.current_controls.yaw = random.uniform(-1.0, 1.0)
            self.current_controls.roll = random.uniform(-1.0, 1.0)
            self.current_controls.jump = random.random() > 0.9
            self.current_controls.boost = random.random() > 0.9
            self.current_controls.handbrake = random.random() > 0.9
            
            # print(f"New Controls: T={self.current_controls.throttle:.2f}, S={self.current_controls.steer:.2f}, J={self.current_controls.jump}, B={self.current_controls.boost}")

        super().step(self.current_controls)

def main():
    print("Starting Random Input Test...")
    
    sim_wrapper = RandomSimWrapper(n_envs=1, max_cars=6)
    
    app = QtWidgets.QApplication([])
    ui.update_scaling_factor(app)

    window = QRSVWindow(QRSVGLWidget(app.primaryScreen(), sim_wrapper))
    window.showNormal()
    app.exec_()

if __name__ == "__main__":
    main()
