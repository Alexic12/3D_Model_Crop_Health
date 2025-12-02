import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from PyQt6.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class NDVIVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.file_path = None
        self.sheets = []
        self.current_sheet_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

    def initUI(self):
        layout = QVBoxLayout()
        
        self.label = QLabel("Upload an Excel file with NDVI data")
        layout.addWidget(self.label)
        
        self.upload_btn = QPushButton("Upload Excel File")
        self.upload_btn.clicked.connect(self.load_file)
        layout.addWidget(self.upload_btn)
        
        self.canvas = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas)
        
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self.toggle_animation)
        layout.addWidget(self.play_pause_btn)
        
        self.setLayout(layout)
    
    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx *.xls)")
        if file_name:
            self.file_path = file_name
            self.sheets = pd.ExcelFile(self.file_path).sheet_names
            self.current_sheet_index = 0
            self.update_plot()
    
    def update_plot(self):
        if not self.file_path:
            return
        
        xls = pd.ExcelFile(self.file_path)
        sheet_name = self.sheets[self.current_sheet_index]
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        
        longitudes = df.iloc[1, 1:].values
        latitudes = df.iloc[2:, 0].values
        ndvi_values = df.iloc[2:, 1:].values
        
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        
        X, Y = np.meshgrid(longitudes, latitudes)
        sc = ax.scatter(X, Y, c=ndvi_values.flatten(), cmap='autumn', vmin=-1, vmax=1)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"NDVI Visualization - {sheet_name}")
        self.canvas.figure.colorbar(sc, ax=ax, label='NDVI Index')
        self.canvas.draw()
        
        self.current_sheet_index = (self.current_sheet_index + 1) % len(self.sheets)
    
    def toggle_animation(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_pause_btn.setText("Play")
        else:
            self.timer.start(1000)  # Update every second
            self.play_pause_btn.setText("Pause")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NDVIVisualizer()
    window.show()
    sys.exit(app.exec())
