import cv2
import numpy as np

class BBoxUtils:
    def __init__(self):
        pass

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox

        # center of x
        x = (x1 + x2) / 2 
        # center of y
        y = (y1 + y2) / 2

        return int(x), int(y)

    def _get_width(self, bbox):
        x1, _, x2, _ = bbox

        # calculate width
        w = x2 - x1

        return int(w)
    
    def _draw_ellipse(self, frame, x_center, y2, width, color):
        # Calculate axes of ellipse
        minor_axis, major_axis = int(width), int(width * 0.35)

        # draw ellipse on the frame
        cv2.ellipse(
            img=frame, 
            center=(x_center, y2),
            axes=(minor_axis, major_axis),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        return frame

    def _draw_rectangle_with_text(self, frame, x_center, y2, track_id, color):
        # define rectangle size
        rectangle_width = 40
        rectangle_height = 20

        # calculate rectangle coordinates
        x1_rectangle = int(x_center - rectangle_width // 2)
        x2_rectangle = int(x_center + rectangle_width // 2)
        y1_rectangle = (y2 - rectangle_height // 2) + 15
        y2_rectangle = (y2 + rectangle_height // 2) + 15

        # draw filled rectangle
        cv2.rectangle(frame,
                      pt1=(x1_rectangle, y1_rectangle),
                      pt2=(x2_rectangle, y2_rectangle),
                      color=color,
                      thickness=cv2.FILLED,
                      lineType=cv2.LINE_4)

        # adjust text position
        x1_text = x1_rectangle + 13
        y1_text = y1_rectangle + 15
        if track_id > 99:
            x1_text -= 5

        # draw track id text
        cv2.putText(frame,
                    str(track_id),
                    org=(x1_text, y1_text),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 0),
                    thickness=2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        # get x and y axis
        y = int(bbox[1])
        x, _ = self._get_center(bbox)

        # declate triangle points
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])

        # draw filled inverted triangle
        cv2.drawContours(frame, 
                         contours=[triangle_points], 
                         contourIdx=0, 
                         color=color, 
                         thickness=cv2.FILLED)

        # add border to triangle
        cv2.drawContours(frame, 
                         contours=[triangle_points], 
                         contourIdx=0, 
                         color=(0, 0, 0), 
                         thickness=2)

        return frame

    def draw_custom_bbox(self, frame, bbox, color, track_id=None):
        # get center of bounding box
        x_center, _ = self._get_center(bbox)
        y2 = int(bbox[3])

        # get width of bounding box
        width = self._get_width(bbox)

        # draw ellipse
        frame = self._draw_ellipse(frame, x_center, y2, width, color)

        # draw rectangle with text if track_id is provided
        if track_id is not None:
            frame = self._draw_rectangle_with_text(frame, x_center, y2, track_id, color)

        return frame
