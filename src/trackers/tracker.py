import pickle
import supervision as sv
from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path):
        # initialize model
        self.model = YOLO(model_path)

        # initialize tracker
        self.tracker = sv.ByteTrack()
    
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
                tracks = pickle.load(stub_path, f)
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

            # now get detections from supervision so we can get tracks using tracker
            sv_detections = sv.Detections.from_ultralytics(detection)

            # convert 'goalkeeper' class_id to 'player' class_id in sv_detections
            # because model is not performing consistent with referee due to small dataset
            for object_idx, class_id in enumerate(sv_detections.class_id):
                if class_names[class_id] == 'goalkeeper':
                    sv_detections.class_id[object_idx] = class_names_inverse['player']

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(sv_detections, frame_num)

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
                    tracks['players'][frame_num][tracker_id] = {'bbox':bbox}
                
                # assign 'referee' bounding box to corresponding tracker_id in current frame
                if class_id == class_names_inverse['referee']:
                    tracks['referees'][frame_num][tracker_id] = {'bbox':bbox}
            
            # we are taking ball without tracks
            for frame_detecion in sv_detections:
                bbox = frame_detection[0].tolist()
                class_id = frame_detecion[3]

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