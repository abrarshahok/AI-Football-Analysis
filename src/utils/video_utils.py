import cv2

class VideoUtils:
    def __init__(self):
        pass

    def read_video(self, video_path):
        # open video file for reading
        capture = cv2.VideoCapture(video_path)
        
        # just return None if video is not opening
        if not capture.isOpened():
            return None
        
        # capture and store frames
        frames = []
        while True:
            # read the frame of video
            ret, frame = capture.read()

            # break if not frames are avaliable
            if not ret:
                break
            
            # store frame
            frames.append(frame)
        
        # close video
        capture.release()

        return frames

    def save_video(self, output_video_frames, output_video_path):
        # define output format of video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # define frame size
        (y, x, c) = output_video_frames[0].shape

        # define VideoWriter for saving frames as video at given path
        output = cv2.VideoWriter(output_video_path, fourcc, 24, (x, y))

        # saving frames as video 
        for frame in output_video_frames:
            output.write(frame)
        
        # close video writer
        output.release()
