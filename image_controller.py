from pubsub import pub
import asyncio
import cv2
import numpy as np 


class ImageController:
    def __init__(self):
        self.bind_events()

    def bind_events(self):
        # This is all the events that this class is listening to
        pub.subscribe(self.handle_upload_image, "upload image")
        pub.subscribe(self.detect_edges , "detect edges")
        pub.subscribe(self.handle_distribution_curve, "draw_distribution")
        pub.subscribe(self.handle_histogram_equalization, "histogram equalization")
        pub.subscribe(self.handle_image_normalizarion, "normalize image")  
    

    def handle_image_normalizarion(self, image):   
        pass

    def handle_histogram_equalization(self, image):
        pass



    def handle_distribution_curve(self, image):
        async def draw_distribution(image):

            await asyncio.sleep(3)

            pub.sendMessage("distribution curve drawn", result=f"this is distribution curve of ({image})")
            print(f"distribution curve drawn for {image}")


    def handle_upload_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            pub.sendMessage("image uploaded", image=image)
        except Exception as e:
            print(e)



    def detect_edges(self, image, filter_type):
            print("Detecting edges")
            # Convert to gray scale
            copyImage = image
            copyImage = cv2.cvtColor(copyImage, cv2.COLOR_BGR2GRAY)
            
            # Convert image to float32 for better precision
            copyImage = copyImage.astype(np.float32)
            
            # Initialize output arrays
            rows, cols = copyImage.shape
            x_edge_image = np.zeros_like(copyImage, dtype=np.float32)
            y_edge_image = np.zeros_like(copyImage, dtype=np.float32)
            filtered_image = np.zeros_like(copyImage, dtype=np.float32)

            # Define kernels
            if filter_type == "Sobel":
                Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
                Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
            elif filter_type == "Perwitt":
                Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
                Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
            elif filter_type == "Robert":
                Kx = np.array([[1, 0], [0, -1]])
                Ky = np.array([[0, 1], [-1, 0]])
            else:
                filtered_image = cv2.Canny(image, 0, 200)
     
                results = {
                    "x_edges" : np.zeros_like(copyImage),
                    "y_edges" : np.zeros_like(copyImage),
                    "filtered_image": filtered_image
                } 
                pub.sendMessage("edges detected", results=results)
                return

            # Optimized loop with numpy operations
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    # Extract 3x3 neighborhood
                    if filter_type == "Robert":
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

            # Normalize the results to 0-255 range
            x_edge_image = cv2.normalize(x_edge_image, None, 0, 255, cv2.NORM_MINMAX)
            y_edge_image = cv2.normalize(y_edge_image, None, 0, 255, cv2.NORM_MINMAX)
            filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)

            # Convert back to uint8
            x_edge_image = x_edge_image.astype(np.uint8)
            y_edge_image = y_edge_image.astype(np.uint8)
            filtered_image = filtered_image.astype(np.uint8)
            
            # Convert original image back to BGR for display
            copyImage = cv2.cvtColor(copyImage.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Return the output as dict
            results = {
                "x_edges": x_edge_image,
                "y_edges": y_edge_image,
                "filtered_image": filtered_image
            }
            
            print("Edges detected")
            pub.sendMessage("edges detected", results=results)
