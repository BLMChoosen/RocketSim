import numpy as np
import os
import cmf_loader
import pickle

# Constants
BT_TO_UU = 50.0
CELL_SIZE = 512.0
MAX_TRIS_PER_CELL = 256  # Adjust if needed

def build_mesh_grid():
    base_path = "collision_meshes/soccar"
    if not os.path.exists(base_path):
        print(f"Path {base_path} not found.")
        return

    meshes = cmf_loader.load_all_meshes(base_path)
    
    all_tris = []
    all_verts = []
    
    # Combine all meshes
    vert_offset = 0
    for i, m in enumerate(meshes):
        # Scale vertices to UU
        verts = np.array(m.vertices) * BT_TO_UU
        tris = np.array(m.tris) + vert_offset
        
        if len(verts) > 0:
            min_v = np.min(verts, axis=0)
            max_v = np.max(verts, axis=0)
            print(f"Mesh {i}: {len(tris)} tris. Bounds: {min_v} to {max_v}")
        
        all_verts.append(verts)
        all_tris.append(tris)
        
        vert_offset += len(verts)
        
    vertices = np.concatenate(all_verts, axis=0)
    triangles = np.concatenate(all_tris, axis=0)
    
    print(f"Total: {len(triangles)} tris, {len(vertices)} verts")
    
    # Compute triangle AABBs
    tri_verts = vertices[triangles] # (N, 3, 3)
    tri_mins = np.min(tri_verts, axis=1) # (N, 3)
    tri_maxs = np.max(tri_verts, axis=1) # (N, 3)
    
    # Define Grid
    # Arena bounds (approx)
    min_bounds = np.array([-4096.0, -5120.0, 0.0])
    max_bounds = np.array([4096.0, 5120.0, 2048.0])
    
    # Add some padding
    min_bounds -= CELL_SIZE
    max_bounds += CELL_SIZE
    
    grid_shape = np.ceil((max_bounds - min_bounds) / CELL_SIZE).astype(int)
    print(f"Grid shape: {grid_shape}")
    
    grid = np.full((*grid_shape, MAX_TRIS_PER_CELL), -1, dtype=np.int32)
    cell_counts = np.zeros(grid_shape, dtype=np.int32)
    
    # Populate Grid
    # This is the slow part in Python, but we only do it once
    print("Populating grid...")
    
    for i in range(len(triangles)):
        # Find grid cells overlapping with tri AABB
        t_min = tri_mins[i]
        t_max = tri_maxs[i]
        
        start_idx = np.floor((t_min - min_bounds) / CELL_SIZE).astype(int)
        end_idx = np.floor((t_max - min_bounds) / CELL_SIZE).astype(int) + 1
        
        # Clamp
        start_idx = np.maximum(start_idx, 0)
        end_idx = np.minimum(end_idx, grid_shape)
        
        for x in range(start_idx[0], end_idx[0]):
            for y in range(start_idx[1], end_idx[1]):
                for z in range(start_idx[2], end_idx[2]):
                    count = cell_counts[x, y, z]
                    if count < MAX_TRIS_PER_CELL:
                        grid[x, y, z, count] = i
                        cell_counts[x, y, z] += 1
                    else:
                        # print(f"Cell {x},{y},{z} full!")
                        pass

    print(f"Max tris in a cell: {np.max(cell_counts)}")
    
    # Pre-compute triangle normals and edges for fast SDF
    # v0, v1, v2
    v0 = tri_verts[:, 0, :]
    v1 = tri_verts[:, 1, :]
    v2 = tri_verts[:, 2, :]
    
    # Edges
    e1 = v1 - v0
    e2 = v2 - v0
    
    # Normals
    normals = np.cross(e1, e2)
    normals /= np.linalg.norm(normals, axis=1)[:, None] + 1e-6
    
    # Save data
    np.savez("mesh_data.npz", 
             grid=grid, 
             tri_verts=tri_verts, 
             tri_normals=normals,
             min_bounds=min_bounds,
             cell_size=CELL_SIZE)
    print("Saved mesh_data.npz")

if __name__ == "__main__":
    build_mesh_grid()
