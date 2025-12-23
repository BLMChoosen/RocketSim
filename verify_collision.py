import jax.numpy as jnp
from collision import arena_sdf

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
    # Plane: 0.707*x + 0.707*y = 5700 (approx)
    # Point on corner: x=4030, y=4030 -> 0.707*8060 = 5700
    test_point("Corner Approx", [4030.0, 4030.0, 1000.0])
    
    # Goal (Inside goal)
    # Y = 5200 (inside back wall hole)
    test_point("Inside Goal", [0.0, 5200.0, 100.0])

if __name__ == "__main__":
    verify()