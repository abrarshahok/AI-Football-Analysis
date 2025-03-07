import cv2
from src.utils import BBoxUtils

class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 5
        self.frame_rate  = 24
        self.bboxUtils = BBoxUtils()

    def add_speed_and_distance_to_tracks(self, tracks):
        # dictionary to store total distance covered by each player
        total_distance = {}  

        for object, object_tracks in tracks.items():
            # process only player tracks
            if object != 'players':  
                continue  

            number_of_frames = len(object_tracks)

            # process every 5th frame to calculate speed
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    # skip if player disappears in the last frame of this batch
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # get player's position at start and end of the frame window
                    start_position = object_tracks[frame_num][track_id]['transformed_position']
                    end_position = object_tracks[last_frame][track_id]['transformed_position']

                      # skip if any position is missing
                    if not start_position or not end_position:
                        continue

                    # measure the distance traveled between the frames
                    distance_covered = self.bboxUtils.measure_distance(start_position, end_position)

                    # calculate time difference in seconds
                    time_elapsed_per_sec = (last_frame - frame_num) / self.frame_rate
                    
                    # compute speed in meters per second and convert to km/h
                    speed_meters_per_sec = distance_covered / time_elapsed_per_sec
                    speed_km_per_hour = speed_meters_per_sec * 3.6  

                    # initialize total distance
                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    # accumulate total distance for the player
                    total_distance[object][track_id] += distance_covered

                    # update speed and distance for each frame in this batch
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        # list to store modified frames with annotations
        output_frames = []  

        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                # process only player tracks
                if object != "players":
                    continue  

                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        
                        if speed is None or distance is None:
                            continue

                        # get bounding box and calculate foot position
                        bbox = track_info['bbox']
                        position = self.bboxUtils.get_foot_position(bbox)  
                        position = list(position)

                        # adjust label position slightly below the player's foot
                        position[1] += 40  
                        # convert to integer tuple
                        position = tuple(map(int, position))  

                        # draw speed and distance on the frame
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames