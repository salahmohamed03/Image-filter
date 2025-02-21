from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QComboBox # Add QFileDialog here
from PyQt6.QtGui import QImage, QPixmap
import cv2
from pubsub import pub
from PyQt6.QtCore import Qt
import numpy as np

class PubSubGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.bind_events()

    def bind_events(self):
        pub.subscribe(self.updateViewer, "fourier calculated")
        pub.subscribe(self.save_image, "image uploaded")
        pub.subscribe(self.displayEdgedImage, "edges detected")


    def initUI(self):
        self.setWindowTitle("PyQt6 Pub-Sub Example")
        self.setGeometry(100, 100, 300, 200)
        layout = QVBoxLayout()
        self.button = QPushButton("calc fourier", self)
        self.button2 = QPushButton("upload image", self)
        self.button3 = QPushButton("detect edges", self)

        # Create a combo box here
        self.combo = QComboBox()
        self.combo.addItem("Sobel")
        self.combo.addItem("Perwitt")
        self.combo.addItem("Robert")
        self.combo.addItem("Canney")
        
        
        self.button.clicked.connect(self.onCalcFourier)
        self.button2.clicked.connect(self.onUploadImage)
        self.button3.clicked.connect(self.onDetectEdges)

        layout.addWidget(self.button)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.combo)
        
        self.setLayout(layout)

        self.label = QLabel("viewer: empty")
        layout.addWidget(self.label)

        # Add Label here to put the image.
        self.imageLabel = QLabel()
        self.edgeLabel = QLabel()
        self.edge2Label = QLabel()

        layout.addWidget(self.imageLabel)
        layout.addWidget(self.edgeLabel)
        layout.addWidget(self.edge2Label)

        self.image=None


    def onCalcFourier(self):
        pub.sendMessage("calc fourier", image="sayed image 1")
        pub.sendMessage("calc fourier", image="sayed image 2")

    def onUploadImage(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
            if file_path:
                pub.sendMessage("upload image", image_path=file_path)
        except Exception as e:
            print(e)

    def save_image(self, image):
        self.image = image
        self.displayImage(image) 

    def displayImage(self, image):
        try:

            height, width, channel = image.shape
            bytes_per_line = 3 * width
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            self.imageLabel.setPixmap(scaled_pixmap)

        except Exception as e:
            print(e)

    def displayEdgedImage(self,results):
        try:
            x_edges, y_edges, filtered_image = results["x_edges"], results["y_edges"], results["filtered_image"]
            print(x_edges)
            
            def convert_to_displayable(edge_img):
                # Ensure the image is uint8
                if edge_img.dtype != np.uint8:
                    edge_img = cv2.normalize(edge_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # If image is grayscale, convert to 3-channel
                if len(edge_img.shape) == 2:
                    edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
                
                return edge_img
            filtered_image = convert_to_displayable(filtered_image)
            self.displayImage(filtered_image)

            if x_edges.all() == 0:
                # Convert both edge images
                x_edges = convert_to_displayable(x_edges)
                y_edges = convert_to_displayable(y_edges)

                

                x_height, x_width, channel = x_edges.shape
                y_height, y_width, channel = y_edges.shape
                x_bytes_per_line = 3 * x_width
                y_bytes_per_line = 3 * y_width

                # Create QImage (no need for BGR2RGB conversion since we already converted to RGB)
                x_qt_image = QImage(x_edges.data, x_width, x_height, x_bytes_per_line, QImage.Format.Format_RGB888)
                y_qt_image = QImage(y_edges.data, y_width, y_height, y_bytes_per_line, QImage.Format.Format_RGB888)

                # Create and scale pixmaps
                x_pixmap = QPixmap.fromImage(x_qt_image)
                y_pixmap = QPixmap.fromImage(y_qt_image)
                x_scaled_pixmap = x_pixmap.scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
                y_scaled_pixmap = y_pixmap.scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)

                # Display the images
                self.edgeLabel.setPixmap(x_scaled_pixmap)
                self.edge2Label.setPixmap(y_scaled_pixmap)

        except Exception as e:
            print(f"Error in displayEdgedImage: {e}")

    def onDetectEdges(self):
        print("Detecting edges")
        pub.sendMessage("detect edges", image=self.image, filter_type=self.combo.currentText())
        

    def updateViewer(self, result):
        self.label.setText(f"viewer: {result}")

