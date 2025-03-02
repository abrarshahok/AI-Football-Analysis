import cv2
import pickle
import numpy as np
from src.utils import BBoxUtils

class CameraMovementEstimator:
    def __init__(self, first_frame):
        # min movement distance threshold
        self.minimum_distance = 5

        # LK optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # convert first frame to grayscale
        first_frame_gray = cv2.cvtColor(first_frame, code=cv2.COLOR_RGB2GRAY)

        # mask to track features only in specific regions
        mask_features = np.zeros_like(first_frame_gray)
        mask_features[:, :20] = 1
        mask_features[:, 900:1050] = 1

        # corner detection params
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

        self.bbox_utils = BBoxUtils()

    def adjust_track_positions(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # get position info
                    position = track_info['position']
                    # get camera movement for current frame
                    camera_movement = camera_movement_per_frame[frame_num]
                    # adjuts position
                    adjusted_position = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    # store adjusted position
                    tracks[object][frame_num][track_id]['adjusted_position'] = adjusted_position

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # load precomputed movements if stub is provided
        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # initialize movement tracking
        camera_movement = [[0, 0]] * len(frames)

        # convert first frame to grayscale and get initial tracking features
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # process each frame
        for frame_num in range(len(frames)):
            new_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # calculate optical flow (track movement)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                prevImg=old_gray, nextImg=new_gray, prevPts=old_features, nextPts=None, **self.lk_params
            )

            # track max movement
            max_distance = 0
            camera_x, camera_y = 0, 0

            # measure movement for each tracked feature
            for i, (new_feature, old_feature) in enumerate(zip(new_features, old_features)):
                # get points (e.g (x, y)) and calculate distance
                new_feature_point = new_feature.ravel()
                old_feature_point = old_feature.ravel()
                distance = self.bbox_utils.measure_distance(new_feature_point, old_feature_point)

                # update max distance and camera movement if needed
                if distance > max_distance:
                    max_distance = distance
                    camera_x, camera_y = self.bbox_utils.measure_xy_distance(new_feature_point, old_feature_point)

            # update movement if it exceeds threshold
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = camera_x, camera_y
                old_features = cv2.goodFeaturesToTrack(new_gray, **self.features)

            # update old frame to new one
            old_gray = new_gray.copy()

        # save computed movements if stub_path is given
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        # initialize output frames
        output_frames = []

        for frame_num, frame in enumerate(frames):
            # create a copy to avoid modifying the original frame
            frame = frame.copy()  

            # create an overlay for displaying movement info
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # extract movement values for current frame
            movement_x, movement_y = camera_movement_per_frame[frame_num]
            movement_x_text = f"Camera Movement X: {movement_x:.2f}"
            movement_y_text = f"Camera Movement Y: {movement_y:.2f}"

            # draw movement text on frame
            frame = cv2.putText(frame, movement_x_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, movement_y_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            # store modified frame
            output_frames.append(frame)  

        return output_frames 