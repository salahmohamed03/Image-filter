from PyQt6.QtWidgets import QMainWindow , QWidget
from PyQt6 import uic
from PyQt6.QtWidgets import QFileDialog
from Messages.Image import Image , Images
from PyQt6.QtGui import QPixmap,QImage
from PyQt6.QtCore import Qt
from pubsub import pub
from Messages.Noise import Noise

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
        self.ui.EdgeDetectionGroupBox.clicked.connect(lambda: self.select_mode(4))
        self.ui.normalizationRadioButton.clicked.connect(self.update_display)
        self.ui.histogramRadioButton.clicked.connect(self.update_display)
        self.ui.thresholdingRadioButton.clicked.connect(self.update_display)
        self.ui.noiseTypeComboBox.currentIndexChanged.connect(self.update_noise)
        self.ui.filterTypeComboBox.currentIndexChanged.connect(self.update_noise)
        self.ui.saltNoiseSlider.valueChanged.connect(self.update_noise)
        self.ui.pepperNoiseSlider.valueChanged.connect(self.update_noise)
        self.ui.edgeDetectionComboBox.currentIndexChanged.connect(self.update_edge_detection)
        self.ui.cutoffFreqOneSlider.valueChanged.connect(self.update_mix_images)
        self.ui.cutoffFreqTwoSlider.valueChanged.connect(self.update_mix_images)

    
    def update_noise(self):
        noise = Noise(
                noise = self.ui.noiseTypeComboBox.currentText(),
                filter = self.ui.filterTypeComboBox.currentText(),
                saltRatio = self.ui.saltNoiseSlider.value()/100,
                pepperRatio = self.ui.pepperNoiseSlider.value()/100
            )
        pub.sendMessage("Add Noise", noise = noise)

    def update_edge_detection(self):
        pub.sendMessage("Edge Detection" , filter = self.ui.edgeDetectionComboBox.currentText())

    def update_mix_images(self):
        pub.sendMessage("Mix Images", freq1 = self.ui.cutoffFreqOneSlider.value(), freq2 = self.ui.cutoffFreqTwoSlider.value())

    def upload_image1(self):
        image = self.upload_image()
        self.images.image1 = image
        size = (250,400)
        if(not self.ui.mixerModeGroup.isChecked()):
            size = (440,400)
        piximage = QPixmap.fromImage(image.qimg.scaled(size[0],size[1]))
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
            self.show_widget(self.ui.out2widget)
            self.ui.outputOneLabel.setText("Image 2")
            self.ui.originalImageLabel.setText("Image 1")
            self.ui.outputTwoLabel.setText("Mixed")
            size = (250,400)
        elif self.ui.EdgeDetectionGroupBox.isChecked():
            self.show_widget(self.ui.out1widget)
            self.show_widget(self.ui.out2widget)
            self.ui.outputOneLabel.setText("Horizontal")
            self.ui.originalImageLabel.setText("Original")
            self.ui.outputTwoLabel.setText("Vertical")
            size = (250,400)
        elif self.ui.otherModesGroup.isChecked():
            self.show_widget(self.ui.out1widget)
            self.show_widget(self.ui.out2widget)
            self.ui.originalImageLabel.setText("Original")
            if self.ui.normalizationRadioButton.isChecked():
                self.ui.outputOneLabel.setText("Normalized")
                self.ui.outputTwoLabel.setText("Equalized")
            elif self.ui.histogramRadioButton.isChecked():
                self.ui.outputOneLabel.setText("Histogram")
                self.ui.outputTwoLabel.setText("CDF")
            elif self.ui.thresholdingRadioButton.isChecked():
                self.ui.outputOneLabel.setText("Local")
                self.ui.outputTwoLabel.setText("Global")

        else:
            self.show_widget(self.ui.out1widget)
            self.ui.originalImageLabel.setText("Original")
            self.hide_widget(self.ui.out2widget)
            self.ui.outputOneLabel.setText("Output")
            size = (440,400)
            
        if self.images.image1:
            image = self.images.image1
            piximage = QPixmap.fromImage(image.qimg.scaled(size[0],size[1]))
            self.ui.OriginalImageLabel.setPixmap(piximage)
            self.ui.OriginalImageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        
        if self.images.output1 and not self.ui.mixerModeGroup.isChecked():
            piximage = QPixmap.fromImage(self.images.output1.qimg.scaled(size[0],size[1]))
            self.ui.OutputImage1Label.setPixmap(piximage)
            self.ui.OutputImage1Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self.images.output2 and self.ui.mixerModeGroup.isChecked():
            piximage = QPixmap.fromImage(self.images.output2.qimg.scaled(size[0],size[1]))
            self.ui.OutputImage2Label.setPixmap(piximage)
            self.ui.OutputImage2Label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def select_mode(self,index):
        if index == 1:
            self.ui.noiseModeGroup.setChecked(False)
            self.ui.otherModesGroup.setChecked(False)
            self.ui.EdgeDetectionGroupBox.setChecked(False)
            self.update_mix_images()
        elif index == 2:
            self.ui.mixerModeGroup.setChecked(False)
            self.ui.otherModesGroup.setChecked(False)
            self.ui.EdgeDetectionGroupBox.setChecked(False)
            noise = Noise(
                noise = self.ui.noiseTypeComboBox.currentText(),
                filter = self.ui.filterTypeComboBox.currentText(),
                saltRatio = self.ui.saltNoiseSlider.value()/100,
                pepperRatio = self.ui.pepperNoiseSlider.value()/100
            )
            self.update_noise()
        elif (index == 3 or ( 
                not self.ui.mixerModeGroup.isChecked() 
                and not self.ui.noiseModeGroup.isChecked()
                and not self.ui.otherModesGroup.isChecked() 
                and not self.ui.EdgeDetectionGroupBox.isChecked())):

            self.ui.otherModesGroup.setChecked(True)
            self.ui.mixerModeGroup.setChecked(False)
            self.ui.noiseModeGroup.setChecked(False)
            self.ui.EdgeDetectionGroupBox.setChecked(False)

            if self.ui.normalizationRadioButton.isChecked():
                pub.sendMessage("Normalize Image")
            elif self.ui.histogramRadioButton.isChecked():
                pub.sendMessage("Histogram Equalization")
            elif self.ui.thresholdingRadioButton.isChecked():
                pub.sendMessage("Thresholding")
        else:
            self.ui.mixerModeGroup.setChecked(False)
            self.ui.noiseModeGroup.setChecked(False)
            self.ui.otherModesGroup.setChecked(False)
            self.update_edge_detection()
