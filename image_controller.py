from io import BytesIO
from pubsub import pub
import asyncio
import cv2
import numpy as np 
from Messages.Image import Image, Images
from PyQt6.QtGui import QImage
import logging
from copy import copy
import concurrent.futures
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg


logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # "w" overwrites the file; use "a" to append
)

class ImageController:
    def __init__(self):
        self.bind_events()

    def bind_events(self):
        # This is all the events that this class is listening to
        
        pub.subscribe(self.handle_distribution_curve, "Histogram Equalization")
        pub.subscribe(self.handel_detect_edges,"Edge Detection")
        pub.subscribe(self.handel_thresholding,"Thresholding")


    
    def handle_distribution_curve(self):
        # Access the image data
        images= Images()
        image_data = images.image1.image_data 

        print("start drawing....")
        
        # red_histo = []
        # green_histo = []
        # blue_histo = []

        fig = plt.figure(figsize=(10, 6))
        plt.title("RGB Color Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        
        # Define colors for plotting
        colors = ('b', 'g', 'r')
        channel_names = ('Blue', 'Green', 'Red')
        
        # Plot histogram for each color channel
        for i, color in enumerate(colors):
            # Calculate histogram for the specific channel
            histogram = cv2.calcHist([image_data], [i], None, [256], [0, 256])
            
            # Normalize histogram for better visualization
            histogram = histogram / histogram.max()
            
            # Plot the histogram with proper label
            plt.plot(histogram, color=color, label=channel_names[i])
        
        # Add a legend to distinguish channels
        plt.legend()
        
        # Set the x-axis limits
        plt.xlim([0, 256])
        
        # Show grid for better readability
        plt.grid(alpha=0.3)
        
        # Show the plot
        plt.tight_layout()
        plt.show()


        # Convert it into image         # Convert Matplotlib plot to QImage
        # buf = BytesIO()
        # plt.savefig(buf, format="png", bbox_inches='tight')
        # plt.close()
        # buf.seek(0)

        # qimage = QImage.fromData(buf.getvalue())


        #images.output1 = self.convert_to_displayable(hist)


        pub.sendMessage("update display")
            

    def handel_thresholding(self):
        print("Debugging thresholding")
        images = Images()
        image = copy(images.image1)
        image_data = copy(image.image_data)
        gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # Apply Local (Adaptive Mean) Thresholding
        local_thresh = cv2.adaptiveThreshold(
           gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Apply Global (Otsu) Thresholding - Extract only the second item (thresholded image)
        _,global_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Normalize the results to 0-255 range
        local_thresh = cv2.normalize(local_thresh, None, 0, 255, cv2.NORM_MINMAX)
        global_thresh = cv2.normalize(global_thresh, None, 0, 255, cv2.NORM_MINMAX)

        # Convert back to uint8
        local_thresh = local_thresh.astype(np.uint8)
        global_thresh = global_thresh.astype(np.uint8)

        # Convert grayscale images to BGR for display
        local_thresh = cv2.cvtColor(local_thresh, cv2.COLOR_GRAY2BGR)
        global_thresh = cv2.cvtColor(global_thresh, cv2.COLOR_GRAY2BGR)

        # Convert images to displayable format
        images.output1 = self.convert_to_displayable(local_thresh) 
        images.output2 = self.convert_to_displayable(global_thresh)  


        logging.info("Update display published from thresholding")
        pub.sendMessage("update display")



    def handle_image_normalizarion(self, image):   
        pass

    def handle_histogram_equalization(self, image):
        pass




    def handle_upload_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            pub.sendMessage("image uploaded", image=image)
        except Exception as e:
            print(e)

    def handel_detect_edges(self, filter):
        pub.sendMessage("start Loading")
        # Create a thread pool executor
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        # Run the CPU-intensive task in a separate thread
        loop.run_in_executor(executor, self.detect_edges_sync, filter)

    def detect_edges_sync(self, filter):
        # Move the content of detect_edges here, without async
        # Remove the async/await keywords
        images = Images()        
        
        # We need to use fft to detect edges
        image = copy(images.image1.image_data)
        copyImage = copy(image)
        copyImage = cv2.cvtColor(copyImage, cv2.COLOR_BGR2GRAY)
        
        # Convert image to float32 for better precision
        copyImage = copyImage.astype(np.float32)
        
        # Initialize output arrays
        rows, cols = copyImage.shape
        x_edge_image = np.zeros_like(copyImage, dtype=np.float32)
        y_edge_image = np.zeros_like(copyImage, dtype=np.float32)
        filtered_image = np.zeros_like(copyImage, dtype=np.float32)

        # Define kernels
        if filter == "Sobel":
            Kx = np.array([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [-1, 0, 1]])
            Ky = np.array([[1, 2, 1], 
                           [0, 0, 0], 
                           [-1, -2, -1]])

        elif filter == "Prewitt":
            Kx = np.array([[-1, 0, 1], 
                           [-1, 0, 1], 
                           [-1, 0, 1]])
            Ky = np.array([[1, 1, 1], 
                           [0, 0, 0], 
                           [-1, -1, -1]])
            
        elif filter == "Roberts":
            Kx = np.array([[1, 0], 
                           [0, -1]])
            Ky = np.array([[0, 1], 
                           [-1, 0]])
        else:
            filtered_image = cv2.Canny(image, 0, 200)
    
            results = {
                "x_edges" : np.zeros_like(copyImage),
                "y_edges" : np.zeros_like(copyImage),
                "filtered_image": filtered_image
            } 
            images.output1 = self.convert_to_displayable(results["x_edges"])
            images.output2 = self.convert_to_displayable(results["y_edges"])
            images.output3 = self.convert_to_displayable(results["filtered_image"])
            logging.info("update display publised from detect edges")
            pub.sendMessage("update display")
            return 
        

        # Optimized loop with numpy operations
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Extract 3x3 neighborhood
                if filter == "Roberts":
                    neighborhood = copyImage[i-1:i+1, j-1:j+1]
                else:
                    neighborhood = copyImage[i-1:i+2, j-1:j+2]
                
                # Calculate gradients using element-wise multiplication and sum
                Gx = np.sum(neighborhood * Kx)
                Gy = np.sum(neighborhood * Ky)
                
                # Store results
                x_edge_image[i, j] = Gx
                y_edge_image[i, j] = Gy
                filtered_image[i, j] = np.sqrt(Gx*Gx + Gy*Gy)

        # # Normalize the results to 0-255 range
        x_edge_image = cv2.normalize(x_edge_image, None, 0, 255, cv2.NORM_MINMAX)
        y_edge_image = cv2.normalize(y_edge_image, None, 0, 255, cv2.NORM_MINMAX)
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)

        # Convert back to uint8
        x_edge_image = x_edge_image.astype(np.uint8)
        y_edge_image = y_edge_image.astype(np.uint8)
        filtered_image = filtered_image.astype(np.uint8)
        
        # Convert original image back to BGR for display
        copyImage = cv2.cvtColor(copyImage.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        filtered_image = cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        images.output1 = self.convert_to_displayable(x_edge_image)
        images.output2 = self.convert_to_displayable(y_edge_image) 
        images.output3 = self.convert_to_displayable(filtered_image)

        logging.info("update display publised from detect edges")
        pub.sendMessage("update display")

    async def detect_edges(self, filter):
        # Keep this as a thin wrapper if needed for backwards compatibility
        return await asyncio.get_event_loop().run_in_executor(None, self.detect_edges_sync, filter)




    @staticmethod
    def convert_to_displayable(edge_img):
        # Ensure the image is uint8
        if edge_img.dtype != np.uint8:
            edge_img = cv2.normalize(edge_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # If image is grayscale, convert to 3-channel
        if len(edge_img.shape) == 2:
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)

        hight, width, channel = edge_img.shape
        bytesPerLine = 3 * width
        edge_img = QImage(edge_img.data, width, hight, bytesPerLine, QImage.Format.Format_RGB888)
        image_data = np.array(edge_img.bits().asarray(edge_img.width()*edge_img.height()*3)).reshape(edge_img.height(),edge_img.width(),3)
        image = Image(image_data=image_data)
        return image
    
    @staticmethod
    def convert_to_displayable_simpel(edge_img):
        # Convert to uint8 if needed and ensure RGB format
        if edge_img.dtype != np.uint8:
            edge_img = cv2.normalize(edge_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(edge_img.shape) == 2:
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
            
        # Create QImage directly from the RGB array
        height, width = edge_img.shape[:2]
        qimage = QImage(edge_img.data, width, height, width * 3, QImage.Format.Format_RGB888)
        
        # Convert to final image format
        image_data = np.frombuffer(qimage.bits(), dtype=np.uint8).reshape(height, width, 3)
        return Image(image_data=image_data)

