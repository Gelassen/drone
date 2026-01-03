# Intent behind this class is to collect a set of handy functions
# to process data without immediate refactoring. 
# After stable contract they should be moved into specific classes  
class Utils:

    # def __init__(self):
        
    def calculate_camera_params(self, frame):
        height, width = frame.shape[:2]
        cx = width / 2
        cy = height / 2
        return [600.0, 600.0, cx, cy]