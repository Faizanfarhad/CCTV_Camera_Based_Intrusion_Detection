import cv2
import numpy as np 
from VideoIngestion.denoise import denoise_frame
from DetectionEngine.motion_detection import foreground_model
from DetectionEngine.background_model import background_model
from DetectionEngine.visualize_polyogn import extract_roi
from DetectionEngine.pretrianed_anomaly_detection import detect_anomaly
from DetectionEngine.motion_detection import MotionDetector
from collections import deque
import os 


class VideoIngestion:
    """_summary_
    Takes vinput as input and returns the video as output.

    vinput : provide camera index(int) if the input is camera itself
            otherwise provide the video path(str)
    
    denoise : method options: "fast", "nlm", "bilateral", "median", "gaussian", "none" 
    
    denoise_strength : 
    """

    def __init__(
        self,
        vinput: int | str,
        denoise: str = "fast",
        denoise_strength: int = 10,
        anamoly_threshold : int = 0.4 
    ):
        super().__init__()
        self.vinput = vinput
        self.denoise = denoise
        self.denoise_strength = denoise_strength
        self.points = []
        self.test_fram = None
        self.fg_model = foreground_model(var_threshold=20,min_threshold=200)
        self.bg_model =  background_model()
        self.previous_frames = deque(maxlen=125)
        self.anomaly_writer = None
        self.video_count = 0
        
        # ROI-based anomaly detection runs per frame on the polygon the user
        # draws during `select_points` (see `run`). The FUVAS video model is a
        # separate, training-required path in pretrianed_anomaly_detection.py.
        
        
        
    def mouseEvent(self,event,x,y,flag,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x,y))
            print(f"points marked at  ({x},{y}) ")
        
    def select_points(self,frame):
        self.test_fram = frame.copy()
        window_name = "ROI Selector"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name,self.mouseEvent)
        
        while True:
            # Stop if the user closes the window with the X button,
            # otherwise waitKey() returns -1 forever and the loop never exits.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("ROI window closed.")
                self.points = []
                break

            temp = frame.copy()
            
            for p in self.points:
                cv2.circle(temp,p,5,(255,0,0),-1)
            
            if len(self.points) > 1:
                cv2.polylines(temp,[np.array(self.points,dtype=np.int32)],False,(255,0,0),2)
            
            cv2.imshow(window_name,temp)
            
            
            button =  cv2.waitKey(1) & 0xFF
            
            if button == ord('s'):
                cv2.destroyAllWindows()
                break
            if button == ord('r'):
                self.points.clear()
            
            if button == ord('q'):
                self.points = []
                break
    
    def draw_bbox(self,detections,frame):
        for det in detections:
            x1,y1,x2,y2 = det['bbox']
            self.bg_model.draw_border(
                frame, top_left=(x1, y1), bottom_right=(x2, y2), thickness=3
            )

    
    def run(self,recorded:bool = True,camera_index:int =0):
        dir_name = "saved_videos"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        
        window_size = (800,400)
        roi_window_size = (600,400)
        
        
        if recorded:
            cap = cv2.VideoCapture(self.vinput)
        else:
            cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Provided path doesn't contain the video or path is incorrect!")
            exit()

        # Use the source video's FPS for saved anomaly clips. Some cameras
        # report 0, so keep a safe fallback in that case.
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = 25.0

        used = False
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if not used: 
                    self.select_points(frame=frame)
                    used = True
                
                #applying denoising
                denoised_frame = denoise_frame(
                    frame, method=self.denoise, strength=self.denoise_strength
                )
                
                ##############buffer 
                self.previous_frames.append(denoised_frame.copy())
                
                
                
                ############## ROI-based anomaly detection
                roi_frame = None
                if len(self.points) >= 3:
                    roi_frame, _ = extract_roi(denoised_frame, self.points)
                    is_anomaly, anomaly_score = detect_anomaly(roi_frame)
                    
                    # Draw the selected polygon and anomaly status on the feed.
                    cv2.polylines(
                        denoised_frame,
                        [np.array(self.points, dtype=np.int32)],
                        True,
                        (255, 0, 0),
                        2,
                    )
                    color = (0, 0, 255) if is_anomaly else (0, 255, 0)
                    cv2.putText(
                        denoised_frame,
                        f"Anomaly: {'YES' if is_anomaly else 'no'} ({anomaly_score:.3f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )
                
                ######## Saving the video if anomaly detected with the buffer 
                if anomaly_score >= 0.2 and self.anomaly_writer is None:
                    H,W  = denoised_frame.shape[:2]
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                    self.anomaly_writer = cv2.VideoWriter(
                        f"{dir_name}/anomaly{self.video_count}.mp4",
                        fourcc,
                        source_fps,
                        (W,H)
                        )
                    
                    for old_frame in self.previous_frames:
                        self.anomaly_writer.write(old_frame)
                elif self.anomaly_writer is not None:
                    self.anomaly_writer.write(denoised_frame)
                #######################
                
                
                ################## person detection section  
                if roi_frame is not None:
                    resized_roi_frame = cv2.resize(roi_frame,roi_window_size)
                    denoised_roi_frame = denoise_frame(
                                        resized_roi_frame, method=self.denoise, strength=self.denoise_strength
                                    )
                    detections, annotated = self.bg_model(denoised_roi_frame)

                    self.draw_bbox(detections,denoised_roi_frame)
                ################################
                
                motion_mask, motion_frame = self.fg_model(denoised_frame)
                
                denoised_frame = cv2.resize(denoised_frame, window_size)
                motion_mask = cv2.resize(motion_mask, window_size)
                motion_frame = cv2.resize(motion_frame, window_size)

                cv2.imshow("Denoised Feed", denoised_frame)
                cv2.imshow("Motion Mask", motion_mask)
                cv2.imshow("Motion In Color", motion_frame)
                if roi_frame is not None:
                    cv2.imshow("ROI", denoised_roi_frame)

                key = cv2.waitKey(25) & 0xFF
                # Exit on 'q' or if the viewer window was closed.
                if key == ord('q') or cv2.getWindowProperty("Denoised Feed", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            # releasing the  resources 
            if self.anomaly_writer is not None:
                self.anomaly_writer.release()
                self.anomaly_writer = None
                self.video_count += 1
            
            cap.release()
            cv2.destroyAllWindows()

path = "testVideo/hit_and_run.mp4"
if __name__ == "__main__":
    from DetectionEngine.visualize_polyogn import show
    # method options: "fast", "nlm", "bilateral", "median", "gaussian", "none"
    video = VideoIngestion(path, denoise="fast")
    video.run(recorded=True,camera_index=0)
    points = video.points
    print(f"Selected Points : {points}")
    # show(video.test_fram,points)

    
