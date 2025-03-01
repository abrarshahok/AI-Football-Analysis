import pickle
import pandas as pd
import supervision as sv
from ultralytics import YOLO
from src.utils import BBoxUtils

class Tracker:
    def __init__(self, model_path):
        # initialize model
        self.model = YOLO(model_path)

        # initialize tracker
        self.tracker = sv.ByteTrack()
        
        # Initialize BBoxUtils
        self.bbox_utils = BBoxUtils()
    
    def detect_frames(self, frames):
        # define batch size
        batch_size = 20

        # define detections array to store detections
        detections = []

        # detect frames on batches instead of whole video at once
        for i in range(0, len(frames), batch_size):
            # get 20 (batch_size) frames
            batch_frames = frames[i: i + batch_size]

            # get predictions in 20 frames
            batch_detections = self.model.predict(batch_frames, conf=0.1)

            # append batch_detecions to detections
            detections += batch_detections
        
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # read from stub then return saved tracks
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # detection frames using YOLO model
        detections = self.detect_frames(frames)

        # initialize tracks to store for players, referees and ball
        tracks = {'players':[], 'referees':[], 'ball':[]}

        # iterate over detctions
        for frame_num, detection in enumerate(detections):

            # get detections names (e.g, 2:player, 3:referee etc)
            class_names = detection.names
            # inverse names like (player:2, referee:3)
            class_names_inverse = {v:k for k, v in class_names.items()}

            # convert YOLO detections to supervision format so tracker (ByteTrack) can use it
            sv_detections = sv.Detections.from_ultralytics(detection)

            # convert 'goalkeeper' class_id to 'player' class_id in sv_detections
            # because model is not performing consistent with referee due to small dataset
            for object_idx, class_id in enumerate(sv_detections.class_id):
                if class_names[class_id] == 'goalkeeper':
                    sv_detections.class_id[object_idx] = class_names_inverse['player']

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(sv_detections)

            # appending dictionary for each object
            # key will be track id which we get from tracker and value will be bounding box
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                # index for bbox = 0, class_id = 3, tracker_id = 4
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                tracker_id = frame_detection[4]

                # assign 'player' bounding box to corresponding tracker_id in current frame
                if class_id == class_names_inverse['player']:
                    tracks['players'][frame_num][tracker_id] = {'bbox': bbox}
                
                # assign 'referee' bounding box to corresponding tracker_id in current frame
                if class_id == class_names_inverse['referee']:
                    tracks['referees'][frame_num][tracker_id] = {'bbox': bbox}
            
            # we are taking ball without tracks
            for frame_detection in sv_detections:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                # just setting 1 as track id for ball
                tracker_id = 1

                # assign 'ball' bounding box to corresponding tracker_id in current frame
                if class_id == class_names_inverse['ball']:
                    tracks['ball'][frame_num][tracker_id] = {'bbox': bbox}
        
        # save tracks if stub_path is provided
        if stub_path is not None: 
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks

    def interpolate_ball_position(self, ball_positions):
        # get bounding box of track 1 in ball positions and convert to list
        ball_positions = [position.get(1, {}).get('bbox', []) for position in ball_positions]

        # convert ball positions to pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate missing values and back fill in case if values are missing from starting frame (e.g. frame 1)
        df_ball_positions = df_ball_positions.interpolate().bfill()

        # convert ball positions to same format and return
        ball_positions = [{1: {'bbox': position}} for position in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        # initialize output video frames to store frames after assigning them ellipse bounding box
        output_video_frames = []

        # iterate through video frames
        for frame_num, frame in enumerate(video_frames):
            # make copy to not change original frame
            frame = frame.copy()

            # get players, referees, and ball info stored in tracks
            players_dict = tracks['players'][frame_num]
            referees_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # change rectangle bounding box to ellipse for all tracks in frame for players
            for track_id, player in players_dict.items():
                bbox = player['bbox']
                color = player.get('team_color', (0, 0, 255))
                frame = self.bbox_utils.draw_custom_bbox(frame, bbox, color, track_id)
                if player.get('has_ball', False):
                    frame = self.bbox_utils.draw_triangle(frame, bbox, (0, 0, 255))
            
            # change rectangle bounding box to ellipse for all tracks in frame for referees
            for _, referee in referees_dict.items():
                bbox = referee['bbox']
                color = (0, 255, 255)
                frame = self.bbox_utils.draw_custom_bbox(frame, bbox, color)
            
            # change rectangle bounding box to filled triangle for all tracks in frame for ball
            for _, ball in ball_dict.items():
                bbox = ball['bbox']
                color = (0, 255, 0)
                frame = self.bbox_utils.draw_triangle(frame, bbox, color)
            
            # draw rectangle and show team controling the ball
            frame = self.bbox_utils.draw_team_ball_control(frame, frame_num, team_ball_control)

            # append new frame to output video frames
            output_video_frames.append(frame)
        
        return output_video_frames