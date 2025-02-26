import PIL.Image
import numpy as np
from scipy import fftpack
import PIL
from pubsub import pub
import asyncio
import concurrent.futures
from Messages.Image import Image,Images
from image_controller import ImageController
import logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # "w" overwrites the file; use "a" to append
)

class ImageFT:
    """
    Class for frequency domain operations on images.
    Provides methods for low-pass filtering, high-pass filtering,
    and mixing images using frequency filters without OpenCV.
    """
    
    def __init__(self):
        """Initialize the ImageFT class."""
        self._bind_events()
    
    def _bind_events(self):
        pub.subscribe(self.handel_mix_images, "Mix Images")
        pub.subscribe(self.handel_freq_filter, "Frequency Filters")
        pub.subscribe(self.handel_grayscale, "Grayscale")

    @staticmethod
    def _resize_to_match(img1, img2):
        """
        Resize both images to match the maximum dimensions and add padding.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            
        Returns:
            tuple: (resized_img1, resized_img2) with padding
        """
        # Get maximum dimensions
        max_height = max(img1.shape[0], img2.shape[0])
        max_width = max(img1.shape[1], img2.shape[1])
        
        # Convert to PIL Images for resizing
        if len(img1.shape) == 3:  # Color images
            pil_img1 = PIL.Image.fromarray(img1.astype(np.uint8))
            pil_img2 = PIL.Image.fromarray(img2.astype(np.uint8))
            # Create padded arrays
            padded1 = np.zeros((max_height, max_width, img1.shape[2]), dtype=np.uint8)
            padded2 = np.zeros((max_height, max_width, img2.shape[2]), dtype=np.uint8)
        else:  # Grayscale images
            pil_img1 = PIL.Image.fromarray(img1.astype(np.uint8))
            pil_img2 = PIL.Image.fromarray(img2.astype(np.uint8))
            # Create padded arrays
            padded1 = np.zeros((max_height, max_width), dtype=np.uint8)
            padded2 = np.zeros((max_height, max_width), dtype=np.uint8)
            
        # Resize and convert back to numpy arrays
        resized1 = np.array(pil_img1.resize((max_width, max_height)))
        resized2 = np.array(pil_img2.resize((max_width, max_height)))
            
        # Copy resized images into padded arrays
        padded1[:resized1.shape[0], :resized1.shape[1]] = resized1
        padded2[:resized2.shape[0], :resized2.shape[1]] = resized2
            
        return padded1, padded2
    
    @staticmethod
    def _create_frequency_mask(shape, cutoff, filter_type="lpf"):
        """
        Create a frequency domain mask for filtering.
        
        Args:
            shape (tuple): Shape of the mask (height, width)
            cutoff (float): Cutoff frequency (normalized 0-1)
            filter_type (str): Either "lpf" for low-pass or "hpf" for high-pass
            
        Returns:
            numpy.ndarray: Frequency domain mask
        """
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        # Create meshgrid for distance calculation
        y, x = np.ogrid[:rows, :cols]
        
        # Calculate distance from center
        center_dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        # Normalize to 0-1 range
        center_dist = center_dist / (np.sqrt(crow**2 + ccol**2))
        
        # Create mask
        mask = np.zeros((rows, cols))
        
        if filter_type == "lpf":
            # Low-pass filter: 1 for frequencies below cutoff, 0 for frequencies above
            mask = center_dist < cutoff
        elif filter_type == "hpf":
            # High-pass filter: 0 for frequencies below cutoff, 1 for frequencies above
            mask = center_dist >= cutoff
        
        return mask.astype(float)
    
    def apply_lpf(self, image, cutoff):
        """
        Apply low-pass filter to an image.
        
        Args:
            image (numpy.ndarray): Input image
            cutoff (float): Cutoff frequency (0-1)
            
        Returns:
            numpy.ndarray: Filtered image
        """
        # Handle multi-channel images
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=float)
            for channel in range(image.shape[2]):
                result[:,:,channel] = self._apply_filter_to_channel(image[:,:,channel], cutoff, "lpf")
        else:
            result = self._apply_filter_to_channel(image, cutoff, "lpf")
            
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_hpf(self, image, cutoff):
        """
        Apply high-pass filter to an image.
        
        Args:
            image (numpy.ndarray): Input image
            cutoff (float): Cutoff frequency (0-1)
            
        Returns:
            numpy.ndarray: Filtered image
        """
        # Handle multi-channel images
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=float)
            for channel in range(image.shape[2]):
                result[:,:,channel] = self._apply_filter_to_channel(image[:,:,channel], cutoff, "hpf")
        else:
            result = self._apply_filter_to_channel(image, cutoff, "hpf")
            
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_filter_to_channel(self, channel, cutoff, filter_type):
        """
        Apply filter to a single image channel.
        
        Args:
            channel (numpy.ndarray): Single channel image
            cutoff (float): Cutoff frequency (0-1)
            filter_type (str): Either "lpf" or "hpf"
            
        Returns:
            numpy.ndarray: Filtered channel
        """
        # Apply 2D FFT
        f_transform = fftpack.fft2(channel)
        f_shift = fftpack.fftshift(f_transform)
        
        # Create and apply mask
        mask = self._create_frequency_mask(channel.shape, cutoff, filter_type)
        f_filtered = f_shift * mask
        
        # Inverse shift and transform
        f_ishift = fftpack.ifftshift(f_filtered)
        img_back = fftpack.ifft2(f_ishift)
        
        # Get the real part and convert back to spatial domain
        img_filtered = np.abs(img_back)
        
        return img_filtered
    
    def _add_padding(self,img1,img2):
        rows1, cols1 = img1.shape
        rows2, cols2 = img2.shape
        rows = max(rows1, rows2)
        cols = max(cols1, cols2)
        padded1 = np.zeros((rows, cols))
        padded2 = np.zeros((rows, cols))
        padded1[:rows1, :cols1] = img1
        padded2[:rows2, :cols2] = img2
        return padded1, padded2
    
    def mix_images(self, img1, img2, cutoff_lpf, cutoff_hpf):
        """
        Mix two images by applying a low-pass filter to the first image
        and a high-pass filter to the second image, then combining them.
        
        Args:
            img1 (numpy.ndarray): First image (will be low-pass filtered)
            img2 (numpy.ndarray): Second image (will be high-pass filtered)
            cutoff (float): Cutoff frequency (0-1)
            
        Returns:
            numpy.ndarray: Mixed image
        """
        cutoff_hpf = cutoff_lpf
        # Ensure images have the same size
        img1_resized,img2_resized = self._resize_to_match(img1, img2)
            
        # Apply filters
        low_passed = self.apply_lpf(img1_resized, cutoff_lpf)
        high_passed = self.apply_hpf(img2_resized, cutoff_hpf)
        
        # Combine images
        if len(img1_resized.shape) == 3:  # Color images
            result = np.zeros_like(low_passed, dtype=float)
            for channel in range(low_passed.shape[2]):
                result[:,:,channel] = low_passed[:,:,channel] + high_passed[:,:,channel]
        else:  # Grayscale images
            result = low_passed + high_passed
            
        return np.clip(result, 0, 255).astype(np.uint8)
        
    def handel_mix_images(self,freq1,freq2):
        # # pub.sendMessage("start Loading")
        # # Create a thread pool executor
        # executor = concurrent.futures.ThreadPoolExecutor()
        # loop = asyncio.get_event_loop()
        # # Run the CPU-intensive task in a separate thread
        # loop.run_in_executor(executor, self.mix_images_sync, freq1, freq2)
        self.mix_images_sync(freq1, freq2)


    def mix_images_sync(self, freq1, freq2):
        if Images().image1 is None or Images().image2 is None:
            return
        image1= Images().image1.image_data
        image2= Images().image2.image_data
        freq1 = freq1/100
        freq2 = freq2/100
        result = self.mix_images(image1, image2, freq1, freq2)
        output = Image(image_data=result)
        Images().output1 = output
        pub.sendMessage("update display")
    def handel_freq_filter(self, cutoff):
        # pub.sendMessage("start Loading")
        # # Create a thread pool executor
        # executor = concurrent.futures.ThreadPoolExecutor()
        # loop = asyncio.get_event_loop()
        # # Run the CPU-intensive task in a separate thread
        # loop.run_in_executor(executor, self.apply_filters, cutoff)
        self.apply_filters(cutoff)

    def apply_filters(self, cutoff):
        logging.info("Applying frequency filters")
        image = Images().image1.image_data
        freq = cutoff/100
        result1 = self.apply_lpf(image, freq)
        result2 = self.apply_hpf(image, freq)
        output1 = Image(image_data=result1)
        output2 = Image(image_data=result2)
        Images().output1 = output1
        Images().output2 = output2
        pub.sendMessage("update display")
    
    def convert_to_grayscale(self, image):
        if len(image.shape) == 2:
            return image  # Already grayscale
            
        # Convert RGB to grayscale using luminosity method
        weights = np.array([0.2989, 0.5870, 0.1140])
        grayscale = np.dot(image[..., :3], weights)
        return grayscale.astype(np.uint8)
    
    def handel_grayscale(self):
        if Images().image1 is not None:
            image = Images().image1.image_data
            result = self.convert_to_grayscale(image)
            output = ImageController().convert_to_displayable(result)
            Images().image1 = output

        if Images().image2 is not None:
            image = Images().image2.image_data
            result = self.convert_to_grayscale(image)
            output = ImageController().convert_to_displayable(result)
            Images().image2 = output

        pub.sendMessage("update display")
    
