"""
Verify arena SDF collision geometry
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
from src.collision import arena_sdf

def test_point(name, pos):
    pos_arr = jnp.array([pos])
    dist, norm = arena_sdf(pos_arr)
    print(f"{name}: Pos={pos}, Dist={dist[0]:.2f}, Normal={norm[0]}")

def verify():
    print("--- Verifying Arena SDF ---")
    # Center (Safe)
    test_point("Center Air", [0.0, 0.0, 1000.0])
    
    # Floor
    test_point("Floor", [0.0, 0.0, 0.0])
    test_point("Below Floor", [0.0, 0.0, -50.0])
    
    # Side Wall (X = 4096)
    test_point("Side Wall", [4096.0, 0.0, 1000.0])
    test_point("Past Side Wall", [4200.0, 0.0, 1000.0])
    
    # Back Wall (Y = 5120)
    test_point("Back Wall", [0.0, 5120.0, 1000.0])
    
    # Corner (45 deg)
    test_point("Corner Approx", [4030.0, 4030.0, 1000.0])
    
    # Goal (Inside goal)
    test_point("Inside Goal", [0.0, 5200.0, 100.0])

if __name__ == "__main__":
    verify()
