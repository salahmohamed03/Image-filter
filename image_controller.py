from pubsub import pub
import asyncio
import cv2
import numpy as np 


class Image:
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
        async def detecting(image, filter_type):
            x_edges_image , y_edges_image, filtered_image = np.zeros_like(image.shape)
            row, col = image.shape
            image = image.astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if filter_type == "Canney":
                filtered_image = cv2.Canny(image, 100, 200)
            await asyncio.sleep(2)

            pub.sendMessage("edges detected", result=f"this is edge of ({image}) with filter {filter_type}")
            print(f"edges detected for {image} with {filter_type}")

