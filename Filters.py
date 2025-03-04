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
    filemode="a" 
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
    width = qimage.width()
    height= qimage.height()
    ptr = qimage.bits()
    bytes_per_line = qimage.bytesPerLine()
    ptr.setsize(height * bytes_per_line)

    bytes_per_line = qimage.bytesPerLine()
    channels = bytes_per_line // width
    print(channels)
    
    try:
        arr = np.array(ptr).reshape(height, bytes_per_line)[:, :width * channels].reshape(height, width, channels)
    except Exception as e:
        print("Reshape error:", e)
        raise

    prob = 0.05
    rnd = np.random.rand(height,width) 
    arr[rnd<prob/2] = 0  
    arr[rnd> 1-prob/2] = 255  

    new_image = QImage(arr.tobytes(), width, height, qimage.bytesPerLine(), qimage.format())
    return new_image

def add_uniform_noise(qimage):
 
    width = qimage.width()
    height = qimage.height()
    
    ptr = qimage.bits()
    ptr.setsize(height * qimage.bytesPerLine())
    
    bytes_per_line = qimage.bytesPerLine()
    try:
        arr = np.array(ptr).reshape(height, bytes_per_line)[:, :width * (bytes_per_line // width)]
        arr = arr.reshape(height, width, bytes_per_line // width)
    except Exception as e:
        print("Reshape error in add_uniform_noise:", e)
        raise
    
    arr_int = arr.astype(np.int16)
    
    noise = np.random.randint(-20, 21, size=arr.shape, dtype=np.int16)
    
    arr_noisy = np.clip(arr_int + noise, 0, 255).astype(np.uint8)
    
    new_image = QImage(arr_noisy.data, width, height, qimage.bytesPerLine(), qimage.format())
    return new_image

def add_gaussian_noise(qimage):
    """
    Adds Gaussian noise to the image.
    The noise has a mean of 0 and a standard deviation of 25.
    """
    width = qimage.width()
    height = qimage.height()
    
    ptr = qimage.bits()
    ptr.setsize(height * qimage.bytesPerLine())
    
    channels = qimage.bytesPerLine() // width
    try:
        arr = np.array(ptr).reshape(height, width, channels)
    except Exception as e:
        print("Reshape error in add_gaussian_noise:", e)
        raise

    arr_int = arr.astype(np.int16)
    
    noise = np.random.normal(0, 25, size=arr.shape)
    
    arr_noisy = np.clip(arr_int + noise, 0, 255).astype(np.uint8)
    
    new_image = QImage(arr_noisy.data, width, height, qimage.bytesPerLine(), qimage.format())
    return new_image




def get_effective_array(qimage):
    width = qimage.width()
    height = qimage.height()
    row_bytes = qimage.bytesPerLine()

    channels = row_bytes // width

    ptr = qimage.bits()
    ptr.setsize(height * row_bytes)

    raw = np.array(ptr, copy=True)

    raw_2d = raw.reshape(height, row_bytes)
    effective_row_bytes = width * channels
    raw_2d = raw_2d[:, :effective_row_bytes]

    arr = raw_2d.reshape(height, width, channels)
    return arr, effective_row_bytes, channels



def create_gaussian_kernel_freq(shape, sigma):
    """Creates a Gaussian kernel in frequency domain"""
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    y = np.fft.fftfreq(rows)[:, None]
    x = np.fft.fftfreq(cols)
    
    y, x = np.meshgrid(y, x, indexing='ij')
    d_squared = x**2 + y**2
    
    return np.exp(-2 * (np.pi**2) * sigma**2 * d_squared)

def apply_gaussian_filter(qimage, sigma=2.0):
    arr, new_row_bytes, channels = get_effective_array(qimage)
    height, width = arr.shape[:2]
    
    kernel_freq = create_gaussian_kernel_freq((height, width), sigma)
    out = np.empty_like(arr)
    
    for c in range(channels):
        img_freq = np.fft.fft2(arr[:,:,c])
        filtered_freq = img_freq * kernel_freq
        out[:,:,c] = np.real(np.fft.ifft2(filtered_freq))
    
    out = np.clip(out, 0, 255).astype(np.uint8)
    new_image = QImage(out.data, width, height, new_row_bytes, qimage.format())
    return new_image

def create_average_kernel_freq(shape, size):
    """Creates an average filter kernel in frequency domain"""
    rows, cols = shape
    spatial_kernel = np.ones((size, size)) / (size * size)
    
    padded_kernel = np.zeros((rows, cols))
    start_row = (rows - size) // 2
    start_col = (cols - size) // 2
    padded_kernel[start_row:start_row+size, start_col:start_col+size] = spatial_kernel
    
    return np.fft.fft2(np.fft.ifftshift(padded_kernel))

def apply_average_filter(qimage, size=7):
    arr, new_row_bytes, channels = get_effective_array(qimage)
    height, width = arr.shape[:2]
    
    if size % 2 == 0:
        size += 1
    
    kernel_freq = create_average_kernel_freq((height, width), size)
    out = np.empty_like(arr)
    
    for c in range(channels):
        img_freq = np.fft.fft2(arr[:,:,c])
        filtered_freq = img_freq * kernel_freq
        out[:,:,c] = np.real(np.fft.ifft2(filtered_freq))
    
    out = np.clip(out, 0, 255).astype(np.uint8)
    new_image = QImage(out.data, width, height, new_row_bytes, qimage.format())
    return new_image


def apply_median_filter(qimage, size=7):
    """
    Applies a median filter with the given window size (default 7x7) using Fourier transform.
    """
    arr, new_row_bytes, channels = get_effective_array(qimage)
    height, width = arr.shape[:2]

    if size % 2 == 0:
        size += 1

    spatial_kernel = np.ones((size, size)) / (size * size)

    padded_kernel = np.zeros((height, width))
    start_row = (height - size) // 2
    start_col = (width - size) // 2
    padded_kernel[start_row:start_row+size, start_col:start_col+size] = spatial_kernel

    kernel_freq = np.fft.fft2(np.fft.ifftshift(padded_kernel))
    out = np.empty_like(arr)

    for c in range(channels):
        img_freq = np.fft.fft2(arr[:, :, c])
        filtered_freq = img_freq * kernel_freq
        out[:, :, c] = np.real(np.fft.ifft2(filtered_freq))

    out = np.clip(out, 0, 255).astype(np.uint8)
    new_image = QImage(out.data, width, height, new_row_bytes, qimage.format())
    return new_image
