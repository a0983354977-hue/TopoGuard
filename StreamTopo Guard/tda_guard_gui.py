import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import threading
import queue
from data_adapter import DataAdapter
from tda_engine import TDAEngine


class TDAGuardGUI:
    """
    TDA-Guard GUI Application with async computation and dual-window visualization.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("TDA-Guard: Topological Anomaly Detection Tool")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.data_adapter = DataAdapter()
        self.tda_engine = TDAEngine()
        
        # Threading
        self.computation_queue = queue.Queue()
        self.is_computing = False
        
        # Results storage
        self.current_data = None
        self.current_diag = None
        self.current_stats = None
        
        # Build GUI
        self._build_ui()
    
    def _build_ui(self):
        """Build the complete user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control Panel (Left)
        control_panel = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # File loading section
        ttk.Label(control_panel, text="Data Source:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.btn_load_csv = ttk.Button(control_panel, text="Load CSV", command=self.load_csv)
        self.btn_load_csv.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.btn_load_json = ttk.Button(control_panel, text="Load JSON", command=self.load_json)
        self.btn_load_json.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.btn_generate_sample = ttk.Button(control_panel, text="Generate Sample Data", command=self.generate_sample_data)
        self.btn_generate_sample.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Separator(control_panel, orient='horizontal').grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Parameters section
        ttk.Label(control_panel, text="Parameters:", font=('Arial', 10, 'bold')).grid(row=5, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(control_panel, text="Max Edge Length:").grid(row=6, column=0, sticky=tk.W)
        self.max_edge_var = tk.DoubleVar(value=1.5)
        ttk.Entry(control_panel, textvariable=self.max_edge_var).grid(row=7, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(control_panel, text="Downsample Size:").grid(row=8, column=0, sticky=tk.W)
        self.downsample_var = tk.IntVar(value=500)
        ttk.Entry(control_panel, textvariable=self.downsample_var).grid(row=9, column=0, sticky=(tk.W, tk.E))
        
        ttk.Separator(control_panel, orient='horizontal').grid(row=10, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Computation section
        ttk.Label(control_panel, text="Computation:", font=('Arial', 10, 'bold')).grid(row=11, column=0, sticky=tk.W, pady=5)
        
        self.btn_compute = ttk.Button(control_panel, text="Compute Topology", command=self.start_computation)
        self.btn_compute.grid(row=12, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_panel, mode='indeterminate')
        self.progress.grid(row=13, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = ttk.Label(control_panel, text="Ready", foreground="blue")
        self.status_label.grid(row=14, column=0, sticky=tk.W, pady=5)
        
        ttk.Separator(control_panel, orient='horizontal').grid(row=15, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Statistics section
        ttk.Label(control_panel, text="Statistics:", font=('Arial', 10, 'bold')).grid(row=16, column=0, sticky=tk.W, pady=5)
        
        self.stats_text = tk.Text(control_panel, height=10, width=30, state='disabled')
        self.stats_text.grid(row=17, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Visualization Panel (Right)
        viz_panel = ttk.Frame(main_frame)
        viz_panel.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        viz_panel.columnconfigure(0, weight=1)
        viz_panel.columnconfigure(1, weight=1)
        viz_panel.rowconfigure(0, weight=1)
        
        # 3D Point Cloud (Left)
        self.fig_3d = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=viz_panel)
        self.canvas_3d.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(viz_panel, text="3D Point Cloud", font=('Arial', 10, 'bold')).grid(row=1, column=0, pady=5)
        
        # Persistence Diagram (Right)
        self.fig_pers = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax_pers = self.fig_pers.add_subplot(111)
        self.canvas_pers = FigureCanvasTkAgg(self.fig_pers, master=viz_panel)
        self.canvas_pers.get_tk_widget().grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(viz_panel, text="Persistence Diagram", font=('Arial', 10, 'bold')).grid(row=1, column=1, pady=5)
    
    def load_csv(self):
        """Load CSV file."""
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            try:
                self.data_adapter.load_csv(filepath)
                self.status_label.config(text=f"Loaded CSV: {self.data_adapter.get_shape()}", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def load_json(self):
        """Load JSON file."""
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filepath:
            try:
                self.data_adapter.load_json(filepath)
                self.status_label.config(text=f"Loaded JSON: {self.data_adapter.get_shape()}", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load JSON: {str(e)}")
    
    def generate_sample_data(self):
        """Generate sample data for testing."""
        try:
            # Generate synthetic data with topological structure
            n_samples = 500
            np.random.seed(42)
            
            # Create a torus-like structure
            theta = np.random.uniform(0, 2*np.pi, n_samples)
            phi = np.random.uniform(0, 2*np.pi, n_samples)
            
            R = 3.0
            r = 1.0
            
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            
            # Add some noise
            noise = np.random.normal(0, 0.1, (n_samples, 3))
            data = np.column_stack([x, y, z]) + noise
            
            self.data_adapter.load_numpy(data)
            self.status_label.config(text=f"Generated sample data: {self.data_adapter.get_shape()}", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate sample data: {str(e)}")
    
    def start_computation(self):
        """Start async computation in background thread."""
        if self.is_computing:
            messagebox.showwarning("Warning", "Computation already in progress")
            return
        
        if self.data_adapter.data is None:
            messagebox.showwarning("Warning", "No data loaded. Please load data first.")
            return
        
        self.is_computing = True
        self.btn_compute.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="Computing topological manifold...", foreground="orange")
        
        # Start background thread
        thread = threading.Thread(target=self._compute_thread)
        thread.daemon = True
        thread.start()
        
        # Schedule callback check
        self.root.after(100, self._check_computation)
    
    def _compute_thread(self):
        """Background thread for TDA computation."""
        try:
            # Get parameters
            max_edge = self.max_edge_var.get()
            downsample_size = self.downsample_var.get()
            
            # Downsample if needed
            data = self.data_adapter.data.copy()
            if data.shape[0] > downsample_size:
                data = self.data_adapter.downsample(target_size=downsample_size)
            
            # Normalize
            normalized_data = self.data_adapter.normalize_zscore(data)
            
            # Compute persistence
            normalized_data, diag = self.tda_engine.compute_persistence_topology(
                normalized_data, 
                max_edge=max_edge
            )
            
            # Compute statistics
            stats = self.tda_engine.get_statistics(diag)
            
            # Put results in queue
            self.computation_queue.put({
                'success': True,
                'data': normalized_data,
                'diag': diag,
                'stats': stats
            })
        except Exception as e:
            self.computation_queue.put({
                'success': False,
                'error': str(e)
            })
    
    def _check_computation(self):
        """Check if computation is complete and update UI safely."""
        try:
            result = self.computation_queue.get_nowait()
            
            if result['success']:
                # Update UI in main thread
                self.current_data = result['data']
                self.current_diag = result['diag']
                self.current_stats = result['stats']
                
                self._update_visualizations()
                self._update_statistics()
                
                self.status_label.config(
                    text=f"Computation complete ({result['stats']['computation_time']:.2f}s)", 
                    foreground="green"
                )
            else:
                messagebox.showerror("Error", f"Computation failed: {result['error']}")
                self.status_label.config(text="Computation failed", foreground="red")
            
            self.is_computing = False
            self.btn_compute.config(state='normal')
            self.progress.stop()
            
        except queue.Empty:
            # Still computing, check again later
            self.root.after(100, self._check_computation)
    
    def _update_visualizations(self):
        """Update 3D point cloud and persistence diagram."""
        # Clear previous plots to prevent memory leaks
        self._clear_canvas(self.ax_3d, self.fig_3d)
        self._clear_canvas(self.ax_pers, self.fig_pers)
        
        # Plot 3D point cloud
        if self.current_data.shape[1] >= 3:
            self.ax_3d.scatter(
                self.current_data[:, 0], 
                self.current_data[:, 1], 
                self.current_data[:, 2],
                c='blue', 
                alpha=0.6,
                s=20
            )
        else:
            # If 2D data, use third dimension as zeros
            self.ax_3d.scatter(
                self.current_data[:, 0], 
                self.current_data[:, 1], 
                np.zeros(self.current_data.shape[0]),
                c='blue', 
                alpha=0.6,
                s=20
            )
        
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('3D Point Cloud')
        
        # Plot persistence diagram
        pairs = self.tda_engine.extract_persistence_pairs(self.current_diag)
        
        # H0 (connected components) - red
        if pairs[0]:
            h0_birth = [b for b, d in pairs[0]]
            h0_death = [d for b, d in pairs[0]]
            self.ax_pers.scatter(h0_birth, h0_death, c='red', label='H0', alpha=0.6, s=30)
        
        # H1 (loops) - blue
        if pairs[1]:
            h1_birth = [b for b, d in pairs[1]]
            h1_death = [d for b, d in pairs[1]]
            self.ax_pers.scatter(h1_birth, h1_death, c='blue', label='H1', alpha=0.6, s=30)
        
        # Plot diagonal line
        max_val = max([max(b, d) for pairs_list in pairs.values() for b, d in pairs_list] + [1])
        self.ax_pers.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        self.ax_pers.set_xlabel('Birth')
        self.ax_pers.set_ylabel('Death')
        self.ax_pers.set_title('Persistence Diagram')
        self.ax_pers.legend()
        self.ax_pers.grid(True, alpha=0.3)
        
        # Refresh canvases
        self.canvas_3d.draw()
        self.canvas_pers.draw()
    
    def _update_statistics(self):
        """Update statistics display."""
        self.stats_text.config(state='normal')
        self.stats_text.delete(1.0, tk.END)
        
        if self.current_stats:
            stats_str = ""
            for key, value in self.current_stats.items():
                if key == 'computation_time':
                    stats_str += f"\nComputation Time: {value:.3f}s\n"
                else:
                    stats_str += f"\n{key}:\n"
                    for stat_key, stat_value in value.items():
                        stats_str += f"  {stat_key}: {stat_value:.4f}\n"
            
            self.stats_text.insert(tk.END, stats_str)
        
        self.stats_text.config(state='disabled')
    
    def _clear_canvas(self, ax, fig):
        """Clear Matplotlib canvas to prevent memory leaks."""
        ax.clear()
        fig.clf()
        fig.add_subplot(ax)


def main():
    root = tk.Tk()
    app = TDAGuardGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
