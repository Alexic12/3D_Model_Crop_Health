import sys
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
from io import BytesIO

# Google Maps API Key (Replace with your key)
API_KEY = "AIzaSyB1Vv2XMsTy1AxEowrzOaI5Sn96ffC6HNY"

class MapViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Google Maps Image Viewer")
        self.setGeometry(100, 100, 600, 600)
        
        # Labels & Input Fields
        self.min_lat_label = QLabel("Min Latitude:")
        self.min_lat_input = QLineEdit("48.8500")  # Example: Paris
        
        self.max_lat_label = QLabel("Max Latitude:")
        self.max_lat_input = QLineEdit("48.8509")  # Approx 1 hectare
        
        self.min_lon_label = QLabel("Min Longitude:")
        self.min_lon_input = QLineEdit("2.3000")
        
        self.max_lon_label = QLabel("Max Longitude:")
        self.max_lon_input = QLineEdit("2.3009")
        
        # Fetch Button
        self.fetch_button = QPushButton("Fetch Map")
        self.fetch_button.clicked.connect(self.fetch_map)
        
        # Image Label
        self.image_label = QLabel("Map will be displayed here")
        self.image_label.setStyleSheet("border: 1px solid black;")
        
        # Layouts
        layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        
        # Adding Widgets
        input_layout.addWidget(self.min_lat_label)
        input_layout.addWidget(self.min_lat_input)
        input_layout.addWidget(self.max_lat_label)
        input_layout.addWidget(self.max_lat_input)
        input_layout.addWidget(self.min_lon_label)
        input_layout.addWidget(self.min_lon_input)
        input_layout.addWidget(self.max_lon_label)
        input_layout.addWidget(self.max_lon_input)
        
        layout.addLayout(input_layout)
        layout.addWidget(self.fetch_button)
        layout.addWidget(self.image_label)
        
        self.setLayout(layout)
    
    def fetch_map(self):
        try:
            min_lat = float(self.min_lat_input.text())
            max_lat = float(self.max_lat_input.text())
            min_lon = float(self.min_lon_input.text())
            max_lon = float(self.max_lon_input.text())
            
            # Calculate center and zoom level (approximate, adjust as needed)
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            zoom = 12  # You can adjust this value
            
            url = f"https://maps.googleapis.com/maps/api/staticmap?center={center_lat},{center_lon}&zoom={zoom}&size=600x400&maptype=satellite&key={API_KEY}"
            
            response = requests.get(url)
            if response.status_code == 200:
                pixmap = QPixmap()
                pixmap.loadFromData(BytesIO(response.content).getvalue())
                self.image_label.setPixmap(pixmap)
            else:
                self.image_label.setText("Error loading map. Check API key and coordinates.")
        except Exception as e:
            self.image_label.setText(f"Error: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MapViewer()
    window.show()
    sys.exit(app.exec_())


    #if idw_file exists

 