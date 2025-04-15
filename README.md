# Image Filter

## Table of Contents
1. [Introduction](#introduction)
2. [Features & Screenshots](#features--screenshots)
   - [Mixer Mode](#-mixer-mode)
   - [Edge Detection](#-edge-detection)
   - [Noise Mode](#-noise-mode)
   - [Filter Mode](#-filter-mode)
   - [Other Modes](#-other-modes)
3. [How to Use](#how-to-use)
4. [Contributors](#contributors)

---

## Introduction
This project implements an **Image Processing Toolbox** using PyQt6, designed to provide a comprehensive suite of tools for image analysis and manipulation. The toolbox integrates **classical algorithms** such as Sobel, Otsu, and Fourier transforms, enabling users to perform tasks like **noise simulation, filtering, edge detection, and histogram analysis**. The **responsive GUI** allows for real-time visualization and multi-image comparison, making it suitable for both **educational purposes** and **practical applications** in fields like computer vision, medical imaging, and digital photography.

---

## Features & Screenshots

### Mixer Mode  
The **Mixer Mode** allows users to blend two images using **frequency-domain manipulation**. By adjusting cutoff sliders, users can control the contribution of low and high-frequency components from each image, enabling advanced image fusion techniques. This mode is particularly useful for applications in **image stitching** and **multi-spectral imaging**.

| Snapshot | Description |
|----------|-------------|
| ![image](https://github.com/user-attachments/assets/1106fcc8-1914-4c0b-a5d9-26160338f037) | Blend two images using frequency-domain manipulation with adjustable cutoff sliders. |

---
### Edge Detection  
The **Edge Detection Mode** implements classical edge detection algorithms, including **Sobel**, **Prewitt**, **Roberts**, and **Canny**. These techniques are fundamental in **feature extraction** and **object detection**, with applications in **autonomous vehicles** and **robotics**.

| Edge Detection Method | Snapshot | Description |
|-----------------------|----------|-------------|
| Sobel | ![image](https://github.com/user-attachments/assets/f2ad30bb-51de-404e-8e2d-facfb8e32959)
 | Detects edges using gradient approximation. |
| Prewitt |![image](https://github.com/user-attachments/assets/c88f2d5d-3a6c-43d7-aef1-93e8194ebdad)
 | Similar to Sobel but with a different kernel. |
| Roberts | ![image](https://github.com/user-attachments/assets/9023efdf-7638-433e-be66-13abf469526d)
 | Detects edges using diagonal gradients. |
| Canny | ![image](https://github.com/user-attachments/assets/ae252113-fbb9-47c5-8b34-35f6e1f5612c)
 | A multi-stage algorithm for optimal edge detection. |

---
### Noise Mode  
The **Noise Mode** simulates various types of noise, including **Gaussian**, **Uniform**, and **Salt & Pepper** noise. This mode is essential for understanding how noise impacts image quality and how different filters can mitigate its effects. It is particularly relevant in **signal processing** and **noise reduction research**.

| Noise Type | Snapshot | Description |
|------------|----------|-------------|
| Gaussian Noise | ![image](https://github.com/user-attachments/assets/06911f84-f888-4182-95c4-2f34cd93ce1e)
 | Simulates additive Gaussian noise, commonly used to model sensor noise. |
| Uniform Noise | ![image](https://github.com/user-attachments/assets/001a5180-c1a4-4455-ab12-f043bd16b66d)
| Simulates uniform noise, often used in statistical simulations. |
| Salt & Pepper Noise | ![image](https://github.com/user-attachments/assets/13d48f82-b14c-41ce-9304-e03cac4bf92e) | Simulates impulsive noise, typical in corrupted images. |

---

### Filter Mode  
The **Filter Mode** provides tools to apply **spatial-domain filters** such as **Average**, **Gaussian**, and **Median** filters. These filters are crucial for **noise reduction** and **image smoothing**, with applications in **medical imaging** and **computer vision**.

| Filter Type | Snapshot | Description |
|-------------|----------|-------------|
| Average Filter | ![Average Filter](https://github.com/user-attachments/assets/c82948b8-fb91-4411-b2cf-1f6c44d98789) | Applies a simple averaging filter to smooth the image. |
| Gaussian Filter | ![Gaussian Filter](https://github.com/user-attachments/assets/42218d6b-8101-4799-8bd3-9f2b6a7aee1d) | Applies a Gaussian kernel for noise reduction while preserving edges. |
| Median Filter | ![Median Filter](https://github.com/user-attachments/assets/d5cd1d6f-a232-4b1e-b9da-93fddfc8416b) | Removes salt and pepper noise by replacing each pixel with the median of its neighborhood. |

---


### Other Modes  
The **Other Modes** section includes advanced tools for **intensity normalization**, **thresholding**, **histogram analysis**, and **frequency filtering**. These tools are essential for **image enhancement** and **feature extraction** in **machine learning** and **computer vision pipelines**.

| Mode | Snapshot | Description |
|------|----------|-------------|
| Normalization | ![Normalization](https://github.com/user-attachments/assets/a78f39c8-446a-4516-b7f3-0d151be6f076) | Normalizes pixel intensities for better contrast. |
| Thresholding | ![Thresholding](https://github.com/user-attachments/assets/ef74dc5c-1aca-488e-a2bd-06b27fb423d0) | Applies binary thresholding to segment images. |
| Histogram & CDF | ![Histogram](https://github.com/user-attachments/assets/499d3d15-2a48-4065-8c89-2938a7b81e6c) | Analyzes image histograms and cumulative distribution functions. |
| Frequency Filtering | ![Filtering](https://github.com/user-attachments/assets/5dfbc25b-0f07-4e80-ad76-c8cb865af36e) | Applies frequency-domain filters for advanced image manipulation. |

---

## How to Use
1. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt

## Contributors

<table>
  <tr>
        <td align="center">
      <a href="https://github.com/salahmohamed03">
        <img src="https://avatars.githubusercontent.com/u/93553073?v=4" width="250px;" alt="Salah Mohamed"/>
        <br />
        <sub><b>Salah Mohamed</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Ayatullah-ahmed" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/125223938?v=" width="250px;" alt="Ayatullah Ahmed"/>
        <br />
        <sub><b>Ayatullah Ahmed</b></sub>
      </a>
    </td>
        <td align="center">
      <a href="https://github.com/Abdelrahman0Sayed">
        <img src="https://avatars.githubusercontent.com/u/113141265?v=4" width="250px;" alt="Abdelrahman Sayed"/>
        <br />
        <sub><b>Abdelrahman Sayed</b></sub>
      </a>
    </td>
        </td>
        <td align="center">
      <a href="https://github.com/AhmeedRaafatt">
        <img src="https://avatars.githubusercontent.com/u/125607744?v=4" width="250px;" alt="Ahmed Raffat"/>
        <br />
        <sub><b>Ahmed Rafaat</b></sub>
      </a>
    </td>
  </tr>
</table>
