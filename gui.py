from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel
from pubsub import pub

class PubSubGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.bind_events()

    def initUI(self):
        self.setWindowTitle("PyQt6 Pub-Sub Example")
        self.setGeometry(100, 100, 300, 200)
        layout = QVBoxLayout()
        self.button = QPushButton("calc fourier", self)
        self.button.clicked.connect(self.onCalcFourier)
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.label = QLabel("viewer: empty")
        layout.addWidget(self.label)

    def bind_events(self):
        pub.subscribe(self.updateViewer, "fourier calculated")


    def onCalcFourier(self):
        pub.sendMessage("calc fourier", image="sayed image 1")
        pub.sendMessage("calc fourier", image="sayed image 2")

    def updateViewer(self, result):
        self.label.setText(f"viewer: {result}")

    

