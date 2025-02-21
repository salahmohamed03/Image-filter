import sys
from PyQt6.QtWidgets import QApplication

from MainWindowUI import MainWindowUI
from fourier import Fourier
import asyncio
from qasync import QEventLoop

async def main():
    # Initialize classes
    fourier = Fourier()

    ###
    
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = MainWindowUI()
    window.show()
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)