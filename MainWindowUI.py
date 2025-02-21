from PyQt6.QtWidgets import QMainWindow , QWidget
from PyQt6 import uic
from PyQt6.QtWidgets import QFileDialog
from Messages.Image import Image , Images
from PyQt6.QtGui import QPixmap,QImage
from PyQt6.QtCore import Qt
from pubsub import pub
import Filters

class MainWindowUI(QMainWindow):
    def __init__(self):
        super(MainWindowUI, self).__init__()
        self.ui = uic.loadUi("design.ui", self)
        self.ui.show()
        self.ui.setWindowTitle("Image Filter")
        self._bind_events()
        self._bind_ui_events()
        self.images = Images()



    def _bind_events(self):
        pub.subscribe(self.update_display, "update display")

    def _bind_ui_events(self):
        self.ui.LoadImageButton.clicked.connect(self.upload_image1)
        self.ui.loadImageTwoButton.clicked.connect(self.upload_image2)
        self.ui.mixerModeGroup.clicked.connect(lambda: self.select_mode(1))
        self.ui.noiseModeGroup.clicked.connect(lambda: self.select_mode(2))
        self.ui.otherModesGroup.clicked.connect(lambda: self.select_mode(3))
        self.ui.noiseTypeComboBox.currentIndexChanged.connect(self.apply_noise_filter)
        self.ui.filterTypeComboBox.currentIndexChanged.connect(self.apply_filter)

    def apply_filter(self, index =None):
        try: 
            filtertype = self.ui.filterTypeComboBox.currentText().strip().lower()
            print(f"Selected filter type: {filtertype}")

            original_Qimg = self.images.image1.qimg

            if filtertype== "gaussian":
                new_qimg = Filters.apply_gaussian_filter(original_Qimg)
            
            elif filtertype == "median filters":
                new_qimg= Filters.apply_median_filter(original_Qimg)

            elif filtertype == "average":
                new_qimg = Filters.apply_average_filter(original_Qimg)
            
            else : 
                print("No matching filter")
                new_qimg = original_Qimg
            
            pixmap = QPixmap.fromImage(new_qimg.scaled(250, 400, Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.OutputImage1Label.setPixmap(pixmap)
            self.ui.OutputImage1Label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        except Exception as e:
            print(f"Error in filter: {e}")

        



    def apply_noise_filter(self, index=None):
        try:
            noise_type = self.ui.noiseTypeComboBox.currentText().strip().lower()
            print(f"Selected noise type: {noise_type}")
            if self.images.image1 is None:
                return 
            
            original_Qimg = self.images.image1.qimg

            if noise_type== "salt & pepper":
                new_qimg = Filters.add_salt_and_pepper_noise(original_Qimg)
            
            elif noise_type == "gaussian":
                new_qimg= Filters.add_gaussian_noise(original_Qimg)

            elif noise_type == "uniform":
                new_qimg = Filters.add_uniform_noise(original_Qimg)
            
            else : 
                print("No matching filter")
                new_qimg = original_Qimg
            
            pixmap = QPixmap.fromImage(new_qimg.scaled(250, 400, Qt.AspectRatioMode.KeepAspectRatio))
            self.ui.OutputImage1Label.setPixmap(pixmap)
            self.ui.OutputImage1Label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        except Exception as e:
            print(f"Error in apply_noise_filter: {e}")

    def upload_image1(self):
        image = self.upload_image()
        self.images.image1 = image
        piximage = QPixmap.fromImage(image.qimg.scaled(250,400))
        self.ui.OriginalImageLabel.setPixmap(piximage)
        self.ui.OriginalImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def upload_image2(self):
        image = self.upload_image()
        self.images.image2 = image
        piximage = QPixmap.fromImage(image.qimg.scaled(250,400))
        self.ui.OutputImage1Label.setPixmap(piximage)
        self.ui.OutputImage1Label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    
    @staticmethod
    def hide_widget(widget):
        widget.hide()
        for child in widget.findChildren(QWidget):
            child.hide()
    def show_widget(self, widget):
        widget.show()
        for child in widget.findChildren(QWidget):
            child.show()

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_name:
            return Image(file_name)
        return None
    
    def update_display(self):
        if self.ui.mixerModeGroup.isChecked():
            self.show_widget(self.ui.out1widget)
            self.ui.outputOneLabel.setText("Image 2")
            self.ui.originalImageLabel.setText("Image 1")
            self.show_widget(self.ui.out2widget)
            self.ui.outputTwoLabel.setText("Mixed Image")
        else:
            self.show_widget(self.ui.out1widget)
            self.ui.originalImageLabel.setText("Original Image")
            self.hide_widget(self.ui.out2widget)
            self.ui.outputOneLabel.setText("Output Image")
        if self.images.output1 and not self.ui.mixerModeGroup.isChecked():
            piximage = QPixmap.fromImage(self.images.output1.qimg.scaled(250,400))
            self.ui.OutputImage1Label.setPixmap(piximage)
            self.ui.OutputImage1Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self.images.output2 and self.ui.mixerModeGroup.isChecked():
            piximage = QPixmap.fromImage(self.images.output2.qimg.scaled(250,400))
            self.ui.OutputImage2Label.setPixmap(piximage)
            self.ui.OutputImage2Label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def select_mode(self,index):
        if index == 1:
            self.ui.noiseModeGroup.setChecked(False)
            self.ui.otherModesGroup.setChecked(False)
        elif index == 2:
            self.ui.mixerModeGroup.setChecked(False)
            self.ui.otherModesGroup.setChecked(False)
        else:
            self.ui.mixerModeGroup.setChecked(False)
            self.ui.noiseModeGroup.setChecked(False)
        # if nothing checked cheick the third one
        if not self.ui.mixerModeGroup.isChecked() and not self.ui.noiseModeGroup.isChecked() and not self.ui.otherModesGroup.isChecked():
            self.ui.otherModesGroup.setChecked(True)
        self.update_display()