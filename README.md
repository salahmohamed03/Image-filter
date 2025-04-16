# Image Filter

![Hero Image](https://github.com/user-attachments/assets/a2db8c0b-1878-4a7c-a507-ea9fa985ce3e)

---

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
| ![Mixer Mode](https://github.com/user-attachments/assets/e069bfc4-9c2c-4921-85dc-3fe384f70a74) | Blend two images using frequency-domain manipulation with adjustable cutoff sliders. |

---
### Edge Detection  
The **Edge Detection Mode** implements classical edge detection algorithms, including **Sobel**, **Prewitt**, **Roberts**, and **Canny**. These techniques are fundamental in **feature extraction** and **object detection**, with applications in **autonomous vehicles** and **robotics**.

| Edge Detection Method | Snapshot | Description |
|-----------------------|----------|-------------|
| Sobel | ![Sobel](https://github.com/user-attachments/assets/2c2cf097-2c6b-4b2f-a537-1a146226693f) | Detects edges using gradient approximation. |
| Prewitt | ![Prewitt](https://github.com/user-attachments/assets/d0018c20-c3a9-4f9e-9e8e-eb85d09d3e7e) | Similar to Sobel but with a different kernel. |
| Roberts | ![Roberts](https://github.com/user-attachments/assets/00bf1454-272f-425e-ab90-df58216e8779) | Detects edges using diagonal gradients. |
| Canny | ![Canny](https://github.com/user-attachments/assets/6a1bd53a-2be6-4287-a774-660863b46946) | A multi-stage algorithm for optimal edge detection. |

---
### Noise Mode  
The **Noise Mode** simulates various types of noise, including **Gaussian**, **Uniform**, and **Salt & Pepper** noise. This mode is essential for understanding how noise impacts image quality and how different filters can mitigate its effects. It is particularly relevant in **signal processing** and **noise reduction research**.

| Noise Type | Snapshot | Description |
|------------|----------|-------------|
| Gaussian Noise | ![Gaussian Noise](https://github.com/user-attachments/assets/0a60be6c-9185-46f1-ae68-f91200659390) | Simulates additive Gaussian noise, commonly used to model sensor noise. |
| Uniform Noise | ![Uniform Noise](https://github.com/user-attachments/assets/879cee6d-67ce-498c-82c8-a69806fe2538) | Simulates uniform noise, often used in statistical simulations. |
| Salt & Pepper Noise | ![Salt & Pepper](https://github.com/user-attachments/assets/79cac0fd-5aa9-4fea-a778-c46471d047d3) | Simulates impulsive noise, typical in corrupted images. |

---

### Filter Mode  
The **Filter Mode** provides tools to apply **spatial-domain filters** such as **Average**, **Gaussian**, and **Median** filters. These filters are crucial for **noise reduction** and **image smoothing**, with applications in **medical imaging** and **computer vision**.

| Filter Type | Snapshot | Description |
|-------------|----------|-------------|
| Average Filter | ![Average Filter](https://github.com/user-attachments/assets/6f0bf661-fe74-4fb3-bdc5-8c3f4ab9e87f) | Applies a simple averaging filter to smooth the image. |
| Gaussian Filter | ![Gaussian Filter](https://github.com/user-attachments/assets/70c2197d-2bd8-4c8e-8cd6-036d8754e73d) | Applies a Gaussian kernel for noise reduction while preserving edges. |
| Median Filter | ![Median Filter](https://github.com/user-attachments/assets/97732206-bf73-48e3-9b72-4c849fed0213) | Removes salt and pepper noise by replacing each pixel with the median of its neighborhood. |

---


### Other Modes  
The **Other Modes** section includes advanced tools for **intensity normalization**, **thresholding**, **histogram analysis**, and **frequency filtering**. These tools are essential for **image enhancement** and **feature extraction** in **machine learning** and **computer vision pipelines**.

| Mode | Snapshot | Description |
|------|----------|-------------|
| Normalization | ![Normalization](https://github.com/user-attachments/assets/082cd1f9-d8ab-4e84-bebb-9392213bc507) | Normalizes pixel intensities for better contrast. |
| Thresholding | ![Thresholding](https://github.com/user-attachments/assets/445a0475-195b-4f64-a523-379390629c8d) | Applies binary thresholding to segment images. |
| Histogram & CDF | ![Histogram](https://github.com/user-attachments/assets/56599d59-fa3c-4d96-9512-2e5caca63a16) | Analyzes image histograms and cumulative distribution functions. |
| Frequency Filtering | ![Filtering](https://github.com/user-attachments/assets/7e2c16dc-c63e-4837-a665-88905e45931e) | Applies frequency-domain filters for advanced image manipulation. |

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
