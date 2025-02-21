
from pubsub import pub
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt6.QtGui import QPixmap
from PyQt6.QtGui import QImage
from Messages.Image import Image, Images
from Messages.Noise import Noise
import numpy as np
from scipy import ndimage
import logging

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # "w" overwrites the file; use "a" to append
)

images = Images()

def bind_events():
    pub.subscribe(apply,"Add Noise")


    

def apply(noise: Noise):
    if images.image1 is None:
        return
    
    if noise.noise == "Salt and Pepper":
        new_image = add_salt_and_pepper_noise(images.image1.qimg)
    elif noise.noise == "Uniform":
        new_image = add_uniform_noise(images.image1.qimg)
    elif noise.noise == "Gaussian":
        new_image = add_gaussian_noise(images.image1.qimg)
    else:
        new_image = images.image1.qimg
    
    if noise.filter == "Gaussian":
        new_image = apply_gaussian_filter(new_image)
    elif noise.filter == "Average":
        new_image = apply_average_filter(new_image)
    elif noise.filter == "Median":
        new_image = apply_median_filter(new_image)
    else:
        print("No filter selected")

    image_data = np.array(new_image.bits().asarray(new_image.width()*new_image.height()*3)).reshape(new_image.height(),new_image.width(),3)
    images.output1 = Image(image_data=image_data)
    pub.sendMessage("update display")
    logging.info(f"Applied noise: {noise.noise} and filter: {noise.filter}")

def add_salt_and_pepper_noise (qimage):
    #add the noise then return the image 
    #first lets convert it to numpy array
    width = qimage.width()
    height= qimage.height()
    #########3
    ptr =   qimage.bits()
    ptr.setsize(height * qimage.bytesPerLine())

    channels = qimage.bytesPerLine() // width
    print(channels)
    
    # Reshape the data accordingly
    try:
        arr = np.array(ptr).reshape(height, width, channels)
    except Exception as e:
        print("Reshape error:", e)
        raise

    # making a mask for pepper and salt
    prob = 0.05
    rnd = np.random.rand(height,width) ######## 
    arr[rnd<prob/2] = 0  # this is the pepper
    arr[rnd> 1-prob/2] = 255  # this is the salt

    #the array is modified , now we make a new qimage from it 
    new_image = QImage(arr.data, width, height, qimage.bytesPerLine(), qimage.format())
    return new_image

def add_uniform_noise(qimage):
    # Noise values are drawn from a uniform distribution between -20 and 20.
 
    width = qimage.width()
    height = qimage.height()
    
    # Get the raw image data and set its size
    ptr = qimage.bits()
    ptr.setsize(height * qimage.bytesPerLine())
    
    # Determine the number of channels dynamically
    channels = qimage.bytesPerLine() // width
    try:
        arr = np.array(ptr).reshape(height, width, channels)
    except Exception as e:
        print("Reshape error in add_uniform_noise:", e)
        raise
    
    # Convert to int16 so that we allow negative values
    arr_int = arr.astype(np.int16)
    
    # uniform noise in the range [-20, 20]
    noise = np.random.randint(-20, 21, size=arr.shape, dtype=np.int16)
    
    # adding the noise and clipping between 0 and 255
    arr_noisy = np.clip(arr_int + noise, 0, 255).astype(np.uint8)
    
    # Create a new QImage from the noisy data
    new_image = QImage(arr_noisy.data, width, height, qimage.bytesPerLine(), qimage.format())
    return new_image

def add_gaussian_noise(qimage):
    """
    Adds Gaussian noise to the image.
    The noise has a mean of 0 and a standard deviation of 25.
    """
    width = qimage.width()
    height = qimage.height()
    
    # Get the raw image data and set its size
    ptr = qimage.bits()
    ptr.setsize(height * qimage.bytesPerLine())
    
    # Determine the number of channels dynamically
    channels = qimage.bytesPerLine() // width
    try:
        arr = np.array(ptr).reshape(height, width, channels)
    except Exception as e:
        print("Reshape error in add_gaussian_noise:", e)
        raise

    arr_int = arr.astype(np.int16)
    
    # Generate Gaussian noise with mean=0 and std=25. so that we used normal random
    noise = np.random.normal(0, 25, size=arr.shape)
    
    # Add noise, then clip and convert back to uint8.
    arr_noisy = np.clip(arr_int + noise, 0, 255).astype(np.uint8)
    
    # Create a new QImage from the noisy data
    new_image = QImage(arr_noisy.data, width, height, qimage.bytesPerLine(), qimage.format())
    return new_image




def get_effective_array(qimage):
    width = qimage.width()
    height = qimage.height()
    row_bytes = qimage.bytesPerLine()

    # e.g. channels = 3 (RGB888) or 4 (ARGB32/RGBA8888)
    # or detect from qimage.format() if possible
    channels = row_bytes // width

    ptr = qimage.bits()
    ptr.setsize(height * row_bytes)

    # Make a copy so the data doesn't vanish after the event loop
    raw = np.array(ptr, copy=True)

    # Reshape as (height, row_bytes) then slice out the real pixel data
    raw_2d = raw.reshape(height, row_bytes)
    effective_row_bytes = width * channels
    raw_2d = raw_2d[:, :effective_row_bytes]

    arr = raw_2d.reshape(height, width, channels)
    return arr, effective_row_bytes, channels



def apply_gaussian_filter(qimage, sigma=2.0):
    #higher sigma for higher blur 
    arr, new_row_bytes, channels = get_effective_array(qimage)
    height, width = arr.shape[:2]

    # Create an output array of the same shape
    out = np.empty_like(arr)

    # Apply Gaussian filter per channel
    for c in range(channels):
        out[:, :, c] = ndimage.gaussian_filter(arr[:, :, c], sigma=sigma, mode='nearest')

    # Clip to [0, 255] in case of floating rounding
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Reconstruct the QImage with the effective row width
    new_image = QImage(out.data, width, height, new_row_bytes, qimage.format())
    return new_image


def apply_average_filter(qimage, size=7):
    """
    Applies a simple average (mean) filter of a given size (default 3x3).
    """
    arr, new_row_bytes, channels = get_effective_array(qimage)
    height, width = arr.shape[:2]

    out = np.empty_like(arr)
    for c in range(channels):
        # Uniform filter is basically an average filter
        out[:, :, c] = ndimage.uniform_filter(arr[:, :, c], size=size, mode='nearest')

    out = np.clip(out, 0, 255).astype(np.uint8)
    new_image = QImage(out.data, width, height, new_row_bytes, qimage.format())
    return new_image


def apply_median_filter(qimage, size=7):
    """
    Applies a median filter with the given window size (default 3x3).
    """
    arr, new_row_bytes, channels = get_effective_array(qimage)
    height, width = arr.shape[:2]

    out = np.empty_like(arr)
    for c in range(channels):
        out[:, :, c] = ndimage.median_filter(arr[:, :, c], size=size, mode='nearest')

    out = np.clip(out, 0, 255).astype(np.uint8)
    new_image = QImage(out.data, width, height, new_row_bytes, qimage.format())
    return new_image
