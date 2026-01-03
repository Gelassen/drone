
class FrameMetadata:

    def __init__(
            self,
            frame,
            camera_params,
            gray,
            tag_size
    ):
        self.frame = frame
        self.camera_params = camera_params
        self.gray = gray
        self.tag_size = tag_size