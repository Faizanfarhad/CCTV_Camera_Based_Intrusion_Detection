import cv2
from typing import Optional, List

from denoise import denoise_frame


class VideoIngestion:
    """_summary_
    Takes vinput as input and returns the video as output.

    vinput : provide camera index(int) if the input is camera itself
             otherwise provide the video path(str)
    """

    def __init__(
        self,
        vinput: int | str,
        denoise: str = "fast",
        denoise_strength: int = 10,
    ):
        super().__init__()
        self.vinput = vinput
        self.denoise = denoise
        self.denoise_strength = denoise_strength

    def run(self):
        cap = cv2.VideoCapture(self.vinput)

        if not cap.isOpened():
            print("Error: Provided path doesn't contain the video or path is incorrect!")
            exit()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1200, 700))
            frame = denoise_frame(
                frame, method=self.denoise, strength=self.denoise_strength
            )

            cv2.imshow("Denoised Feed", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


path = "testVideo/animal_video.mp4"
if __name__ == "__main__":
    # method options: "fast", "nlm", "bilateral", "median", "gaussian", "none"
    video = VideoIngestion(path, denoise="fast")
    video.run()

#TODO : Configurable detection zones (draw ROI polygon on frame). 