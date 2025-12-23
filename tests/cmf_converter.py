import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import struct
import os
import glob
import threading

class CMFConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RL Collision Converter (.cmf -> .obj) [FIXED]")
        self.root.geometry("700x500")
        self.root.resizable(False, False)

        # Variables
        self.source_path = tk.StringVar()
        self.dest_path = tk.StringVar()

        # UI Layout
        self.create_widgets()

    def create_widgets(self):
        # --- Source Folder Selection ---
        lbl_source = tk.Label(self.root, text="Source Folder (contains .cmf files):", font=("Arial", 10, "bold"))
        lbl_source.pack(anchor="w", padx=10, pady=(10, 0))

        frame_source = tk.Frame(self.root)
        frame_source.pack(fill="x", padx=10, pady=5)

        entry_source = tk.Entry(frame_source, textvariable=self.source_path, state="readonly", width=60)
        entry_source.pack(side="left", fill="x", expand=True)

        btn_browse_source = tk.Button(frame_source, text="Browse...", command=self.select_source_folder)
        btn_browse_source.pack(side="right", padx=(5, 0))

        # --- Destination Folder Selection ---
        lbl_dest = tk.Label(self.root, text="Destination Folder (save .obj files here):", font=("Arial", 10, "bold"))
        lbl_dest.pack(anchor="w", padx=10, pady=(10, 0))

        frame_dest = tk.Frame(self.root)
        frame_dest.pack(fill="x", padx=10, pady=5)

        entry_dest = tk.Entry(frame_dest, textvariable=self.dest_path, state="readonly", width=60)
        entry_dest.pack(side="left", fill="x", expand=True)

        btn_browse_dest = tk.Button(frame_dest, text="Browse...", command=self.select_dest_folder)
        btn_browse_dest.pack(side="right", padx=(5, 0))

        # --- Action Button ---
        self.btn_convert = tk.Button(self.root, text="START CONVERSION", bg="#2196F3", fg="white", font=("Arial", 11, "bold"), command=self.start_thread)
        self.btn_convert.pack(pady=20, ipadx=20, ipady=5)

        # --- Log Area ---
        lbl_log = tk.Label(self.root, text="Conversion Log:", font=("Arial", 9))
        lbl_log.pack(anchor="w", padx=10)

        self.log_area = scrolledtext.ScrolledText(self.root, width=80, height=15, state="disabled", font=("Consolas", 9))
        self.log_area.pack(padx=10, pady=(0, 10))

    def select_source_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with .cmf files")
        if folder:
            self.source_path.set(folder)
            if not self.dest_path.get():
                self.dest_path.set(folder)

    def select_dest_folder(self):
        folder = filedialog.askdirectory(title="Select Destination Folder")
        if folder:
            self.dest_path.set(folder)

    def log_message(self, message):
        """Thread-safe logging to text widget"""
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state="disabled")

    def start_thread(self):
        source = self.source_path.get()
        dest = self.dest_path.get()

        if not source or not dest:
            messagebox.showwarning("Missing Paths", "Please select both Source and Destination folders.")
            return

        self.btn_convert.config(state="disabled", text="Converting...", bg="#B0BEC5")
        threading.Thread(target=self.run_conversion, args=(source, dest), daemon=True).start()

    def run_conversion(self, source_dir, dest_dir):
        self.log_message(f"--- Started Conversion ---")
        
        cmf_files = glob.glob(os.path.join(source_dir, "*.cmf"))
        
        if not cmf_files:
            self.log_message("ERROR: No .cmf files found in source folder.")
            self.restore_ui()
            return

        self.log_message(f"Found {len(cmf_files)} files.")
        
        success_count = 0
        error_count = 0

        for cmf_file in cmf_files:
            filename = os.path.basename(cmf_file)
            obj_filename = filename.replace(".cmf", ".obj")
            obj_path = os.path.join(dest_dir, obj_filename)

            try:
                # Chama a função de conversão corrigida
                counts = self.convert_file_fixed(cmf_file, obj_path)
                self.log_message(f"[OK] {filename} -> {counts['verts']} verts, {counts['tris']} tris")
                success_count += 1
            except Exception as e:
                self.log_message(f"[ERR] Failed {filename}: {str(e)}")
                error_count += 1

        self.log_message("-" * 30)
        self.log_message(f"DONE. Success: {success_count}, Errors: {error_count}")
        messagebox.showinfo("Done", f"Success: {success_count}\nErrors: {error_count}")
        
        self.restore_ui()

    def restore_ui(self):
        self.root.after(0, lambda: self.btn_convert.config(state="normal", text="START CONVERSION", bg="#2196F3"))

    def convert_file_fixed(self, cmf_path, obj_path):
        """
        Lógica baseada em RocketSim/src/CollisionMeshFile.cpp
        Header: int32 NumVerts, int32 NumTris
        Data: [Vertices...], [Triangles...]
        """
        file_size = os.path.getsize(cmf_path)
        
        with open(cmf_path, "rb") as f:
            data = f.read()

        offset = 0
        
        # 1. HEADER: Lê os dois contadores (8 bytes)
        if len(data) < 8:
            raise ValueError("File too small for header")
            
        # Tenta ler NumVerts e NumTris
        # '<II' significa: Little Endian, Unsigned Int, Unsigned Int
        num_verts, num_tris = struct.unpack_from('<II', data, offset)
        offset += 8
        
        # Validação de sanidade: Verifica se o tamanho do arquivo bate com o esperado
        expected_size = 8 + (num_verts * 12) + (num_tris * 12)
        
        # Nota: Às vezes tem bytes extras no final (hash), então verificamos se tem PELO MENOS o tamanho esperado
        if file_size < expected_size:
             # Tenta inverter a ordem se falhar (caso o formato seja Tris, Verts)
            num_tris_alt, num_verts_alt = num_verts, num_tris
            expected_alt = 8 + (num_verts_alt * 12) + (num_tris_alt * 12)
            
            if file_size >= expected_alt:
                # Era invertido mesmo
                num_verts = num_verts_alt
                num_tris = num_tris_alt
            else:
                raise ValueError(f"Corrupt file or wrong format. Expected >{expected_size} bytes, got {file_size}")

        vertices = []
        # 2. BLOCO DE VÉRTICES
        for _ in range(num_verts):
            vx, vy, vz = struct.unpack_from('<fff', data, offset)
            vertices.append((vx, vy, vz))
            offset += 12 # 3 floats * 4 bytes
        
        indices = []
        # 3. BLOCO DE TRIÂNGULOS (Índices)
        for _ in range(num_tris):
            i1, i2, i3 = struct.unpack_from('<III', data, offset) # RocketSim usa int32 para indices
            # OBJ começa em 1, C++ começa em 0
            indices.append((i1 + 1, i2 + 1, i3 + 1))
            offset += 12 # 3 ints * 4 bytes

        # 4. Escreve .OBJ
        with open(obj_path, "w") as f:
            f.write(f"# RocketSim Dump\n")
            f.write(f"# Verts: {num_verts}, Tris: {num_tris}\n")
            for v in vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for idx in indices:
                f.write(f"f {idx[0]} {idx[1]} {idx[2]}\n")
                
        return {"verts": num_verts, "tris": num_tris}

if __name__ == "__main__":
    root = tk.Tk()
    app = CMFConverterApp(root)
    root.mainloop()