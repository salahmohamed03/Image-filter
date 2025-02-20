from pubsub import pub 
import asyncio


class Fourier:
    def __init__(self):
        self.bind_events()

    def bind_events(self):
        pub.subscribe(self.handle_calc_fourier, "calc fourier")

    def handle_calc_fourier(self, image):
        async def calcFourier(image):
            ###
            # do some calculations
            await asyncio.sleep(3)
            ###
            pub.sendMessage("fourier calculated", result=f"this is fourier of ({image})")
            print(f"fourier calculated for {image}")



            
        loop = asyncio.get_event_loop()
        loop.create_task(calcFourier(image))






