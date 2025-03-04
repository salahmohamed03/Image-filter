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
)

class ImageController:
    def __init__(self):
        self.bind_events()

    def bind_events(self):
        pub.subscribe(self.handle_equalization, "Normalize Image")
        pub.subscribe(self.handle_distribution_curve, "Histogram Equalization")
        pub.subscribe(self.handle_detect_edges,"Edge Detection")
        pub.subscribe(self.handle_thresholding,"Thresholding")


    
    def handle_distribution_curve(self):
        try:
            images = Images()
            image_data = images.image1.image_data 
                        
            if image_data is None or len(image_data.shape) < 3:
                print("Invalid image data")
                return
                
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
            fig.patch.set_facecolor('#0b192c')

            colors = ("b", "g", "r")
            color_channels = ("Blue", "Green", "Red")

            for i, (color, channel_name) in enumerate(zip(colors, color_channels)):
                histogram = self.calculate_histogram(image_data[:, :, i])
                cdf = np.cumsum(histogram)
                cdf_min = cdf.min()
                cdf_max = cdf.max()
                cdf = ((cdf - cdf_min) / (cdf_max - cdf_min) * 255).astype(np.uint8)
                
                for ax in axes[i]:
                    ax.set_facecolor("#0b192c")
                    ax.tick_params(axis='x', colors="white")
                    ax.tick_params(axis='y', colors="white")
                    ax.spines["bottom"].set_color("white")
                    ax.spines["left"].set_color("white")

                axes[i, 0].fill_between(range(256), histogram, color=color, alpha=0.4)
                axes[i, 0].plot(histogram, color=color, linewidth=1.5)
                axes[i, 0].set_title(f"{channel_name} Histogram", color="white")
                axes[i, 0].set_xlabel("Pixel Intensity", color="white")
                axes[i, 0].set_ylabel("Normalized Frequency", color="white")
                axes[i, 0].set_xlim([0, 256])
                axes[i, 0].grid(alpha=0.3, color="gray")

                axes[i, 1].fill_between(range(256), cdf, color=color, alpha=0.4)
                axes[i, 1].plot(cdf, color=color, linewidth=1.5)
                axes[i, 1].set_title(f"{channel_name} CDF", color="white")
                axes[i, 1].set_xlabel("Pixel Intensity", color="white")
                axes[i, 1].set_ylabel("Cumulative Frequency", color="white")
                axes[i, 1].set_xlim([0, 256])
                axes[i, 1].grid(alpha=0.3, color="gray")

            plt.tight_layout()
            fig.subplots_adjust(top=0.92)

            canvas = FigureCanvasAgg(fig)
            pub.sendMessage("display_histogram", canvas=canvas)

        
        except Exception as e:
            print(f"Error in handle_distribution_curve: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def handle_equalization(self):
        images = Images()
        image_data = images.image1.image_data
        image_data =cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        
        histogram = self.calculate_histogram(image_data)
        cdf = np.cumsum(histogram)
        cdf_min = cdf.min()
        cdf_max = cdf.max()
        normalized_cdf = ((cdf - cdf_min) / (cdf_max - cdf_min) * 255).astype(np.uint8)
        

        equalized_image = normalized_cdf[image_data]
        equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
        
        normalized_image = self.normalize_image(image_data)
        images.output1 = self.convert_to_displayable(normalized_image)
        images.output2 = self.convert_to_displayable(equalized_image)
        
        logging.info("Update display published from equalization")
        pub.sendMessage("update display")

    
    def calculate_histogram(self, image_data):
        histogram = np.zeros(256, dtype=np.int32)
        for pixel in image_data.flatten():
                histogram[pixel] += 1
        return histogram
    
    def normalize_image(self, image_data):
        normalized_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        return normalized_image

    def handle_thresholding(self):
        print("Debugging thresholding")
        images = Images()
        image = copy(images.image1)
        image_data = copy(image.image_data)
        gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        local_thresh = cv2.adaptiveThreshold(
           gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )

        _,global_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        local_thresh = cv2.normalize(local_thresh, None, 0, 255, cv2.NORM_MINMAX)
        global_thresh = cv2.normalize(global_thresh, None, 0, 255, cv2.NORM_MINMAX)

        local_thresh = local_thresh.astype(np.uint8)
        global_thresh = global_thresh.astype(np.uint8)

        local_thresh = cv2.cvtColor(local_thresh, cv2.COLOR_GRAY2BGR)
        global_thresh = cv2.cvtColor(global_thresh, cv2.COLOR_GRAY2BGR)

        images.output1 = self.convert_to_displayable(local_thresh) 
        images.output2 = self.convert_to_displayable(global_thresh)  


        logging.info("Update display published from thresholding")
        pub.sendMessage("update display")

    
    def handle_detect_edges(self, filter):
        pub.sendMessage("start Loading")
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, self.detect_edges_sync, filter)


    def fourier_transform(self, image):
        print("FOURIER TRANSFORM STARTED")

        if image is None or image.size == 0:
            print("ERROR: Image is empty or None")
            return None

        image = image.astype(np.float32)

        try:
            ft_components = np.fft.fft2(image)
            ft_components = np.fft.fftshift(ft_components) 
            ft_magnitude = np.log(np.abs(ft_components) + 1)  
            ft_phase = np.angle(ft_components)

            results = {
                "ft_magnitude": ft_magnitude,
                "ft_phase": ft_phase,
                "ft_components": ft_components  
            }

            print("FOURIER TRANSFORM COMPLETE")
            return results
        
        except Exception as e:
            print(f"ERROR in FFT2: {e}")
            return None


    def detect_edges_sync(self, filter):
        images = Images()
        image = copy(images.image1.image_data)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        rows, cols = gray.shape
        print(f"Image shape: {rows}x{cols}")

        # Compute Fourier Transform
        ft_data = self.fourier_transform(gray)
        if ft_data is None:
            print("Error: Fourier transform failed.")
            return
        
        ft_components = ft_data["ft_components"] 

        # Choose Edge Detection Filter
        if filter == "Sobel":
            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel X
            Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Sobel Y
        elif filter == "Prewitt":
            Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        elif filter == "Roberts":
            Kx = np.array([[1, 0], [0, -1]])
            Ky = np.array([[0, 1], [-1, 0]])
        else:
            print("Applying Canny filter instead")

            # Apply Gaussian Blur to separate horizontal/vertical edges
            blurred_x = cv2.GaussianBlur(gray, (1, 5), 0)  # Blur in Y direction
            blurred_y = cv2.GaussianBlur(gray, (5, 1), 0)  # Blur in X direction

            x_edge_image = cv2.Canny(blurred_x.astype(np.uint8), 50, 150)
            y_edge_image = cv2.Canny(blurred_y.astype(np.uint8), 50, 150)
            filtered_image = cv2.Canny(gray.astype(np.uint8), 50, 150)

            # Display results
            results = {
                "x_edges": x_edge_image,
                "y_edges": y_edge_image,
                "filtered_image": filtered_image
            }
            images.output1 = self.convert_to_displayable(results["x_edges"])
            images.output2 = self.convert_to_displayable(results["y_edges"])
            images.output3 = self.convert_to_displayable(results["filtered_image"])
            pub.sendMessage("update display")
            return 

        # Zero-Pad Kernels for Fourier Domain Convolution
        Kx_padded = np.zeros_like(gray)
        Ky_padded = np.zeros_like(gray)

        kh, kw = Kx.shape
        Kx_padded[:kh, :kw] = Kx
        Ky_padded[:kh, :kw] = Ky

        # Compute FFT of the Kernels
        Kx_fft = np.fft.fft2(Kx_padded)
        Ky_fft = np.fft.fft2(Ky_padded)

        # Shift for Proper Convolution
        Kx_fft = np.fft.fftshift(Kx_fft)
        Ky_fft = np.fft.fftshift(Ky_fft)

        # Apply Edge Detection in Fourier Domain
        Gx_fft = ft_components * Kx_fft
        Gy_fft = ft_components * Ky_fft

        x_edge_image = np.fft.ifft2(Gx_fft).real
        y_edge_image = np.fft.ifft2(Gy_fft).real
        filtered_image = np.sqrt(x_edge_image**2 + y_edge_image**2)

        # **Normalize Edge Maps to Enhance Visibility**
        x_edge_image = cv2.normalize(np.abs(x_edge_image), None, 0, 255, cv2.NORM_MINMAX)
        y_edge_image = cv2.normalize(np.abs(y_edge_image), None, 0, 255, cv2.NORM_MINMAX)
        filtered_image = cv2.normalize(np.sqrt(x_edge_image**2 + y_edge_image**2), None, 0, 255, cv2.NORM_MINMAX)


        print("Filtering complete")

        # Store and Display Results
        results = {
            "x_edges": x_edge_image,
            "y_edges": y_edge_image,
            "filtered_image": filtered_image
        }
        images.output1 = self.convert_to_displayable(results["x_edges"])
        images.output2 = self.convert_to_displayable(results["y_edges"])
        images.output3 = self.convert_to_displayable(results["filtered_image"])

        pub.sendMessage("update display")
        print("Edge detection complete, results published")





    @staticmethod
    def convert_to_displayable(edge_img):
        if edge_img.dtype != np.uint8:
            edge_img = cv2.normalize(edge_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
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
        if edge_img.dtype != np.uint8:
            edge_img = cv2.normalize(edge_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(edge_img.shape) == 2:
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
            
        height, width = edge_img.shape[:2]
        qimage = QImage(edge_img.data, width, height, width * 3, QImage.Format.Format_RGB888)
        
        image_data = np.frombuffer(qimage.bits(), dtype=np.uint8).reshape(height, width, 3)
        return Image(image_data=image_data)

