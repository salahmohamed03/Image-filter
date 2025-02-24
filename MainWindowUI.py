from PyQt6.QtWidgets import QMainWindow , QWidget
from PyQt6 import uic
from PyQt6.QtWidgets import QFileDialog
from Messages.Image import Image , Images
from PyQt6.QtGui import QPixmap,QImage
from PyQt6.QtCore import Qt, QTimer
from pubsub import pub
from Messages.Noise import Noise
import logging


logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # "w" overwrites the file; use "a" to append
)

class MainWindowUI(QMainWindow):
    def __init__(self):
        super(MainWindowUI, self).__init__()
        self.ui = uic.loadUi("design.ui", self)
        self.ui.show()
        self.ui.setWindowTitle("Image Filter")
        self.isLoading = False
        self.hide_Widget(self.ui.saltPepperWidget)
        self.hide_Widget(self.ui.LoadingLabel)
        self._bind_events()
        self._bind_ui_events()
        self.images = Images()
        self.images.image1 = Image("test_images/Lenna.png")
        self.update_output()



    def _bind_events(self):
        pub.subscribe(self.update_display, "update display")
        pub.subscribe(self.end_loading, "update display")
        pub.subscribe(self.start_loading, "start Loading")
    def start_loading(self):
        self.show_widget(self.ui.LoadingLabel)
        self.isLoading = True
        logging.info("Loading started")
        
    def end_loading(self):
        self.hide_Widget(self.ui.LoadingLabel)
        self.isLoading = False
        logging.info("Loading ended")

    def _bind_ui_events(self):
        self.ui.LoadImageButton.clicked.connect(self.upload_image1)
        self.ui.loadImageTwoButton.clicked.connect(self.upload_image2)
        self.ui.mixerModeGroup.clicked.connect(lambda: self.select_mode(1))
        self.ui.noiseModeGroup.clicked.connect(lambda: self.select_mode(2))
        self.ui.otherModesGroup.clicked.connect(lambda: self.select_mode(3))
        self.ui.EdgeDetectionGroupBox.clicked.connect(lambda: self.select_mode(4))
        self.ui.normalizationRadioButton.clicked.connect(self.update_display)
        self.ui.histogramRadioButton.clicked.connect(self.update_display)
        self.ui.thresholdingRadioButton.clicked.connect(self.update_thresholding)
        self.ui.noiseTypeComboBox.currentIndexChanged.connect(self.update_noise)
        self.ui.filterTypeComboBox.currentIndexChanged.connect(self.update_noise)
        self.ui.saltNoiseSlider.valueChanged.connect(self.update_noise)
        self.ui.pepperNoiseSlider.valueChanged.connect(self.update_noise)
        self.ui.edgeDetectionComboBox.currentIndexChanged.connect(self.update_edge_detection)
        self.ui.cutoffFreqOneSlider.valueChanged.connect(self.update_mix_images)
        self.ui.cutoffFreqTwoSlider.valueChanged.connect(self.update_mix_images)
        

    def update_output(self):
        if self.ui.mixerModeGroup.isChecked():
            self.select_mode(1)
        elif self.ui.noiseModeGroup.isChecked():
            self.select_mode(2)
        elif self.ui.otherModesGroup.isChecked():
            self.select_mode(3)
        elif self.ui.EdgeDetectionGroupBox.isChecked():
            self.select_mode(4)
        else:
            self.select_mode(0)

    def update_noise(self):
        if self.isLoading:
            QTimer.singleShot(100, self.update_noise)
            return
        noise = Noise(
                noise = self.ui.noiseTypeComboBox.currentText(),
                filter = self.ui.filterTypeComboBox.currentText(),
                saltRatio = self.ui.saltNoiseSlider.value()/100,
                pepperRatio = self.ui.pepperNoiseSlider.value()/100
            )
        pub.sendMessage("Add Noise", noise = noise)
        logging.info(f"Add Noise topic published Noise: {noise.noise} Filter: {noise.filter} Salt: {noise.saltRatio} Pepper: {noise.pepperRatio}")

    def update_edge_detection(self):
        if self.isLoading:
            QTimer.singleShot(100, self.update_edge_detection)
            return
        pub.sendMessage("Edge Detection" , filter = self.ui.edgeDetectionComboBox.currentText())
        logging.info(f"Edge Detection topic published Filter: {self.ui.edgeDetectionComboBox.currentText()}")
    
    def update_thresholding(self):
        if self.isLoading:
            QTimer.singleShot(100, self.update_edge_detection)
            return
        if self.images.image1:  
            pub.sendMessage("Thresholding", image=self.images.image1)  
            logging.info("Thresholding topic published")


    def update_mix_images(self):
        if self.isLoading:
            QTimer.singleShot(100, self.update_mix_images)
            return
        pub.sendMessage("Mix Images", freq1 = self.ui.cutoffFreqOneSlider.value(), freq2 = self.ui.cutoffFreqTwoSlider.value())
        logging.info(f"Mix Images topic published Freq1: {self.ui.cutoffFreqOneSlider.value()} Freq2: {self.ui.cutoffFreqTwoSlider.value()}")

    def upload_image1(self):
        image = self.upload_image()
        self.images.image1 = image
        size = (250,400)
        if(not self.ui.mixerModeGroup.isChecked()):
            size = (440,400)
        piximage = QPixmap.fromImage(image.qimg.scaled(size[0],size[1]))
        self.ui.OriginalImage1Label.setPixmap(piximage)
        self.ui.OriginalImage1Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def upload_image2(self):
        image = self.upload_image()
        self.images.image2 = image
        piximage = QPixmap.fromImage(image.qimg.scaled(250,400))
        self.ui.OriginalImage2Label.setPixmap(piximage)
        self.ui.OriginalImage2Label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    
    @staticmethod
    def hide_Widget(Widget):
        Widget.hide()
        for child in Widget.findChildren(QWidget):
            child.hide()
    def show_widget(self, Widget):
        Widget.show()
        for child in Widget.findChildren(QWidget):
            child.show()

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_name:
            return Image(file_name)
        return None
    
    def update_display(self):
        if self.ui.mixerModeGroup.isChecked():
            self.show_widget(self.ui.Image2Widget)
            self.hide_Widget(self.ui.Out2Widget)
            self.hide_Widget(self.ui.Out3Widget)
            self.ui.OriginalImage1Text.setText("Image 1")
            self.ui.OriginalImage2Text.setText("Image 2")
            self.ui.OutputImage1Text.setText("Mixed")
            size = (250,400)
        elif self.ui.EdgeDetectionGroupBox.isChecked():
            self.show_widget(self.ui.Out2Widget)
            self.show_widget(self.ui.Out3Widget)
            self.hide_Widget(self.ui.Image2Widget)
            self.ui.OriginalImage1Text.setText("Original")
            self.ui.OutputImage2Text.setText("Horizontal")
            self.ui.OutputImage1Text.setText("Vertical")
            self.ui.OutputImage3Text.setText("Edge Detected")
            size = (250,400)
        elif self.ui.otherModesGroup.isChecked():
            self.show_widget(self.ui.Out1Widget)
            self.show_widget(self.ui.Out2Widget)
            self.hide_Widget(self.ui.Out3Widget)
            self.hide_Widget(self.ui.Image2Widget)
            self.ui.OriginalImage1Text.setText("Original")
            if self.ui.normalizationRadioButton.isChecked():
                self.ui.OutputImage1Text.setText("Normalized")
                self.ui.OutputImage2Text.setText("Equalized")
            elif self.ui.histogramRadioButton.isChecked():
                self.ui.OutputImage1Text.setText("Histogram")
                self.ui.OutputImage2Text.setText("CDF")
            elif self.ui.thresholdingRadioButton.isChecked():
                self.ui.OutputImage1Text.setText("Local")
                self.ui.OutputImage2Text.setText("Global")

            size = (250,400)

        else:
            self.hide_Widget(self.ui.Out2Widget)
            self.hide_Widget(self.ui.Out3Widget)
            self.hide_Widget(self.ui.Image2Widget)
            self.ui.OriginalImage1Text.setText("Original")
            self.ui.OutputImage1Text.setText("Output")
            size = (440,400)
            
        if self.images.image1:
            image = self.images.image1
            piximage = QPixmap.fromImage(image.qimg.scaled(size[0],size[1]))
            self.ui.OriginalImage1Label.setPixmap(piximage)
            self.ui.OriginalImage1Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if self.images.image2:
            image = self.images.image2
            piximage = QPixmap.fromImage(image.qimg.scaled(250,400))
            self.ui.OriginalImage2Label.setPixmap(piximage)
            self.ui.OriginalImage2Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
        if self.images.output1:
            piximage = QPixmap.fromImage(self.images.output1.qimg.scaled(size[0],size[1]))
            self.ui.OutputImage1Label.setPixmap(piximage)
            self.ui.OutputImage1Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self.images.output2:
            piximage = QPixmap.fromImage(self.images.output2.qimg.scaled(size[0],size[1]))
            self.ui.OutputImage2Label.setPixmap(piximage)
            self.ui.OutputImage2Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if self.images.output3:
            piximage = QPixmap.fromImage(self.images.output3.qimg.scaled(size[0],size[1]))
            self.ui.OutputImage3Label.setPixmap(piximage)
            self.ui.OutputImage3Label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def select_mode(self,index):
        self.OutputImage1Label.clear()
        self.OutputImage2Label.clear()
        self.OutputImage3Label.clear()
        self.images.output1 = None
        self.images.output2 = None
        self.images.output3 = None
        if index == 1:
            self.ui.noiseModeGroup.setChecked(False)
            self.ui.otherModesGroup.setChecked(False)
            self.ui.EdgeDetectionGroupBox.setChecked(False)
            self.update_mix_images()
        elif index == 2:
            self.ui.mixerModeGroup.setChecked(False)
            self.ui.otherModesGroup.setChecked(False)
            self.ui.EdgeDetectionGroupBox.setChecked(False)
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
                logging.info("Normalize Image topic published")
            elif self.ui.histogramRadioButton.isChecked():
                pub.sendMessage("Histogram Equalization")
                logging.info("Histogram Equalization topic published")
            elif self.ui.thresholdingRadioButton.isChecked():
                self.update_thresholding()


        else:
            self.ui.mixerModeGroup.setChecked(False)
            self.ui.noiseModeGroup.setChecked(False)
            self.ui.otherModesGroup.setChecked(False)
            self.update_edge_detection()
        self.update_display()

