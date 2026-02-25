import struct
import os

class CollisionMeshFile:
    def __init__(self):
        self.tris = []
        self.vertices = []
        self.hash = 0

    def read_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            # Read numTris and numVertices
            data = f.read(8)
            num_tris, num_vertices = struct.unpack('ii', data)
            
            print(f"Loading {file_path}: {num_tris} tris, {num_vertices} verts")

            # Read triangles
            # Each triangle is 3 ints (vertex indexes)
            tri_data = f.read(num_tris * 3 * 4)
            self.tris = list(struct.iter_unpack('iii', tri_data))

            # Read vertices
            # Each vertex is 3 floats (x, y, z)
            vert_data = f.read(num_vertices * 3 * 4)
            self.vertices = list(struct.iter_unpack('fff', vert_data))

            # Read hash (optional, might not be there if file ends?)
            # The C++ code calculates hash, doesn't seem to read it from file?
            # Wait, C++ code: UpdateHash() is called after reading.
            # So hash is not in the file.
            
            return self

def load_all_meshes(base_path):
    meshes = []
    for i in range(16): # mesh_0 to mesh_15
        file_path = os.path.join(base_path, f"mesh_{i}.cmf")
        if os.path.exists(file_path):
            cmf = CollisionMeshFile()
            cmf.read_from_file(file_path)
            meshes.append(cmf)
    return meshes

if __name__ == "__main__":
    base_path = "collision_meshes/soccar"
    if os.path.exists(base_path):
        meshes = load_all_meshes(base_path)
        total_tris = sum(len(m.tris) for m in meshes)
        total_verts = sum(len(m.vertices) for m in meshes)
        print(f"Total: {total_tris} tris, {total_verts} verts")
        
        # Inspect first triangle of first mesh
        if meshes and meshes[0].tris:
            t = meshes[0].tris[0]
            v0 = meshes[0].vertices[t[0]]
            v1 = meshes[0].vertices[t[1]]
            v2 = meshes[0].vertices[t[2]]
            print(f"First Tri: {v0}, {v1}, {v2}")
            
            # Compute normal
            import numpy as np
            p0 = np.array(v0)
            p1 = np.array(v1)
            p2 = np.array(v2)
            normal = np.cross(p1 - p0, p2 - p0)
            normal = normal / np.linalg.norm(normal)
            print(f"Normal: {normal}")
    else:
        print(f"Path {base_path} not found.")
