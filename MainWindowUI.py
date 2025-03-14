from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PyQt6 import uic
from PyQt6.QtWidgets import QFileDialog
from Messages.Image import Image , Images
from PyQt6.QtGui import QPixmap,QImage
from PyQt6.QtCore import Qt, QTimer
from pubsub import pub
from Messages.Noise import Noise
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
        self.histogram = None
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
        pub.subscribe(self.update_histogram, "display_histogram")
        pub.subscribe(self.update_output, "update_output")
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
        # self.ui.cutoffFreqTwoSlider.valueChanged.connect(self.update_mix_images)
        self.ui.normalizationRadioButton.clicked.connect(self.update_output)
        self.ui.histogramRadioButton.clicked.connect(self.update_output)
        self.ui.thresholdingRadioButton.clicked.connect(self.update_output)
        self.ui.freqDomainRadioButton.clicked.connect(self.update_output)
        self.ui.cutoffFrequencySlider.valueChanged.connect(self.update_freq_domain)
        self.ui.resetButton.clicked.connect(self.reset_images)
        self.ui.grayScaleButton.clicked.connect(self.convert_to_grayscale)

    def update_histogram(self, canvas: FigureCanvasAgg):
        self.histogram = FigureCanvas(canvas.figure)
        layout = self.ui.imageDisplayContainer.layout()
        if layout is None:
            layout = QVBoxLayout(self.ui.imageDisplayContainer)
        layout.addWidget(self.histogram)
        print("Histogram displayed")

    def remove_histogram(self):
        layout = self.ui.imageDisplayContainer.layout()
        if layout is not None:
            if self.histogram:
                layout.removeWidget(self.histogram)
                self.histogram.deleteLater()
                self.histogram = None
                print("Histogram removed")


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
    def update_freq_domain(self):
        if Images().image1 is None:
            return
        if not self.ui.freqDomainRadioButton.isChecked():
            return
        pub.sendMessage("Frequency Filters", cutoff = self.ui.cutoffFrequencySlider.value())
        logging.info("Frequency Domain topic published")

    def convert_to_grayscale(self):
        if Images().image1 is None:
            return
        pub.sendMessage("Grayscale")
        logging.info("Grayscale topic published")

    def reset_images(self):
        self.images.image1 = None
        self.images.image2 = None
        self.images.output1 = None
        self.images.output2 = None
        self.images.output3 = None
        self.OriginalImage1Label.clear()
        self.OriginalImage2Label.clear()
        self.OutputImage1Label.clear()
        self.OutputImage2Label.clear()
        self.OutputImage3Label.clear()
        logging.info("Images reset")

    def update_noise(self):
        if Images().image1 is None:
            return
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
        if Images().image1 is None:
            return
        if self.isLoading:
            QTimer.singleShot(100, self.update_edge_detection)
            return
        pub.sendMessage("Edge Detection" , filter = self.ui.edgeDetectionComboBox.currentText())
        logging.info(f"Edge Detection topic published Filter: {self.ui.edgeDetectionComboBox.currentText()}")
    
    def update_thresholding(self):
        if Images().image1 is None:
            return
        if self.isLoading:
            QTimer.singleShot(100, self.update_thresholding)
            return
        if self.images.image1:  
            pub.sendMessage("Thresholding")  
            logging.info("Thresholding topic published")


    def update_mix_images(self):
        if Images().image1 is None:
            return
        if self.isLoading:
            QTimer.singleShot(100, self.update_mix_images)
            return
        pub.sendMessage("Mix Images", freq1 = self.ui.cutoffFreqOneSlider.value(), freq2 = 0)
        logging.info(f"Mix Images topic published Freq1: {self.ui.cutoffFreqOneSlider.value()} Freq2: {0}")

    def upload_image1(self):
        image = self.upload_image()
        if image is None: return
        self.images.image1 = image
        size = (250,400)
        if(not self.ui.mixerModeGroup.isChecked()):
            size = (440,400)
        image.resize((512, 512))
        piximage = QPixmap.fromImage(image.qimg.scaled(size[0],size[1]))
        self.ui.OriginalImage1Label.setPixmap(piximage)
        self.ui.OriginalImage1Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_output()
    
    def upload_image2(self):
        image = self.upload_image()
        if image is None: return
        self.images.image2 = image
        image.resize((512, 512))
        piximage = QPixmap.fromImage(image.qimg.scaled(250,400))
        self.ui.OriginalImage2Label.setPixmap(piximage)
        self.ui.OriginalImage2Label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_output()

    
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
        # self.remove_histogram()
        if self.ui.mixerModeGroup.isChecked():
            self.show_widget(self.ui.Image2Widget)
            self.show_widget(self.ui.Out1Widget)
            self.hide_Widget(self.ui.Out2Widget)
            self.hide_Widget(self.ui.Out3Widget)
            self.ui.OriginalImage1Text.setText("Image 1")
            self.ui.OriginalImage2Text.setText("Image 2")
            self.ui.OutputImage1Text.setText("Mixed")
            size = (250,400)
        elif self.ui.EdgeDetectionGroupBox.isChecked():
            self.show_widget(self.ui.Out1Widget)
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
                self.hide_Widget(self.ui.Out1Widget)
                self.hide_Widget(self.ui.Out2Widget)
                self.hide_Widget(self.ui.Out3Widget)

                # self.ui.OutputImage1Text.setText("Histogram")
                # self.ui.OutputImage2Text.setText("CDF")
            elif self.ui.thresholdingRadioButton.isChecked():
                self.ui.OutputImage1Text.setText("Local")
                self.ui.OutputImage2Text.setText("Global")
            elif self.freqDomainRadioButton.isChecked():
                self.ui.OutputImage1Text.setText("Low Pass")
                self.ui.OutputImage2Text.setText("High Pass")
            size = (250,400)
        else:
            self.show_widget(self.ui.Out1Widget)
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
        self.remove_histogram()
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
            if Images().image1 is None:
                return
            if self.ui.normalizationRadioButton.isChecked():
                pub.sendMessage("Normalize Image")
                logging.info("Normalize Image topic published")
            elif self.ui.histogramRadioButton.isChecked():
                pub.sendMessage("Histogram Equalization")
                logging.info("Histogram Equalization topic published")
            elif self.ui.thresholdingRadioButton.isChecked():
                pub.sendMessage("Thresholding")
                logging.info("Thresholding topic published")
            elif self.freqDomainRadioButton.isChecked():
                pub.sendMessage("Frequency Filters", cutoff = self.ui.cutoffFrequencySlider.value())
                logging.info("Frequency Domain topic published")
        else:
            self.ui.mixerModeGroup.setChecked(False)
            self.ui.noiseModeGroup.setChecked(False)
            self.ui.otherModesGroup.setChecked(False)
            self.update_edge_detection()
        self.update_display()

