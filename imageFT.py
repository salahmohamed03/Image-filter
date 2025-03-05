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
    filemode="a" 
)

class ImageFT:
    
    def __init__(self):
        self._bind_events()
    
    def _bind_events(self):
        pub.subscribe(self.handel_mix_images, "Mix Images")
        pub.subscribe(self.handel_freq_filter, "Frequency Filters")
        pub.subscribe(self.handel_grayscale, "Grayscale")

    @staticmethod
    def _resize_to_match(img1, img2):
        max_height = max(img1.shape[0], img2.shape[0])
        max_width = max(img1.shape[1], img2.shape[1])
        
        if len(img1.shape) == 3: 
            pil_img1 = PIL.Image.fromarray(img1.astype(np.uint8))
            pil_img2 = PIL.Image.fromarray(img2.astype(np.uint8))
            padded1 = np.zeros((max_height, max_width, img1.shape[2]), dtype=np.uint8)
            padded2 = np.zeros((max_height, max_width, img2.shape[2]), dtype=np.uint8)
        else: 
            pil_img1 = PIL.Image.fromarray(img1.astype(np.uint8))
            pil_img2 = PIL.Image.fromarray(img2.astype(np.uint8))
            padded1 = np.zeros((max_height, max_width), dtype=np.uint8)
            padded2 = np.zeros((max_height, max_width), dtype=np.uint8)
            
        resized1 = np.array(pil_img1.resize((max_width, max_height)))
        resized2 = np.array(pil_img2.resize((max_width, max_height)))
            
        padded1[:resized1.shape[0], :resized1.shape[1]] = resized1
        padded2[:resized2.shape[0], :resized2.shape[1]] = resized2
            
        return padded1, padded2
    
    @staticmethod
    def _create_frequency_mask(shape, cutoff, filter_type="lpf"):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        y, x = np.ogrid[:rows, :cols]
        
        center_dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        center_dist = center_dist / (np.sqrt(crow**2 + ccol**2))
        
        mask = np.zeros((rows, cols))
        
        if filter_type == "lpf":
            mask = center_dist < cutoff
        elif filter_type == "hpf":
            mask = center_dist >= cutoff
        
        return mask.astype(float)
    
    def apply_lpf(self, image, cutoff):
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=float)
            for channel in range(image.shape[2]):
                result[:,:,channel] = self._apply_filter_to_channel(image[:,:,channel], cutoff, "lpf")
        else:
            result = self._apply_filter_to_channel(image, cutoff, "lpf")
            
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_hpf(self, image, cutoff):
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=float)
            for channel in range(image.shape[2]):
                result[:,:,channel] = self._apply_filter_to_channel(image[:,:,channel], cutoff, "hpf")
        else:
            result = self._apply_filter_to_channel(image, cutoff, "hpf")
            
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_filter_to_channel(self, channel, cutoff, filter_type):
        key = hash(channel.tobytes())
        if key in Images().cache:
            f_shift = Images().cache[key]
        else:
            f_transform = fftpack.fft2(channel)
            f_shift = fftpack.fftshift(f_transform) # Shift zero frequency component to center
            Images().cache[key] = f_shift
        
        mask = self._create_frequency_mask(channel.shape, cutoff, filter_type)
        f_filtered = f_shift * mask
        
        f_ishift = fftpack.ifftshift(f_filtered)
        img_back = fftpack.ifft2(f_ishift)
        
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
        cutoff_hpf = cutoff_lpf
        img1_resized,img2_resized = self._resize_to_match(img1, img2)
            
        low_passed = self.apply_lpf(img1_resized, cutoff_lpf)
        high_passed = self.apply_hpf(img2_resized, cutoff_hpf)
        
        if len(img1_resized.shape) == 3:
            result = np.zeros_like(low_passed, dtype=float)
            for channel in range(low_passed.shape[2]):
                result[:,:,channel] = low_passed[:,:,channel] + high_passed[:,:,channel]
        else:  
            result = low_passed + high_passed
            
        return np.clip(result, 0, 255).astype(np.uint8)
        
    def handel_mix_images(self,freq1,freq2):
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
            return image  
            
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
        pub.sendMessage("update_output")
    
