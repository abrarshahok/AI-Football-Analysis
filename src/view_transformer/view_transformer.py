import cv2
import numpy as np

class ViewTransformer:
    def __init__(self):
        # real-world dimensions of the football court in meters
        football_court_width = 68 
        football_court_length = 23.32

        # pixel coordinates of key points in the image (e.g., corners of the field)
        self.pixel_vertices = np.array([
            [110, 1035],   # bottom-left
            [265, 275],    # top-left
            [910, 260],    # top-right
            [1640, 915],   # bottom-right
        ]).astype(np.float32)

        # corresponding real-world coordinates (meters)
        self.target_vertices = np.array([
            [0, football_court_width],     # bottom-left
            [0, 0],                        # top-lef
            [football_court_length, 0],    # top-right
            [football_court_length, football_court_width],  # bottom-right
        ]).astype(np.float32)

        # compute the transformation matrix to map pixel coordinates to real-world coordinates
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
    
    def transform_point(self, point):
        # convert to float32 (required for opencv)
        point = np.array(point).astype(np.float32)  
        p = tuple(point)

        # ensure polygon is in the correct shape for opencv
        polygon = self.pixel_vertices.reshape((-1, 1, 2)).astype(np.float32)

        # check if the point is inside the field
        is_inside = cv2.pointPolygonTest(polygon, p, False) >= 0
        if not is_inside:
            return None  

        # reshape the point into the required format for transformation
        reshaped_point = point.reshape(-1, 1, 2)

        # apply perspective transformation
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)

        # reshape the result to a 2D format
        transformed_point = transformed_point.reshape(-1, 2)

        return transformed_point


    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # get the adjusted position of the object
                    position = track_info['adjusted_position']

                    # transform the position to real-world coordinates
                    transformed_position = self.transform_point(position)

                    if transformed_position is not None:
                        transformed_position = np.squeeze(transformed_position).tolist()
                    
                    # save the transformed position back in the tracking data
                    tracks[object][frame_num][track_id]['transformed_position'] = transformed_position