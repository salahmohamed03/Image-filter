

class Filters:
    def __init__():
        self.bind_events()

    def bind_events(self):
        pub.subscribe(self.handle_calc_filter, "upload image")

    
    def handle_calc_filter(self, image):
        ###
        pub.sendMessage("filter calculated", result=f"this is filter of ({image})")
        