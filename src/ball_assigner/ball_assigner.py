from src.utils import BBoxUtils

class BallAssigner:
    def __init__(self):
        self.bbox_utils = BBoxUtils()
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        # get center of ball; means its current position in frame
        ball_position = self.bbox_utils.get_center(ball_bbox)

        # declare min distance and assigned player
        min_distance    = 99999
        assigned_player = -1


        for player_id, player in players.items():
            # get player bounding box
            player_bbox = player['bbox']
            x1, y1, x2, y2 = player_bbox

            # calculate minimum distance
            distance_left = self.bbox_utils.measure_distance((x1, y2), ball_position)
            distance_right = self.bbox_utils.measure_distance((x2, y2), ball_position)
            distance = min(distance_left, distance_right)

            # set player id as current assigned player if distance is less than max distance
            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id
        
        return assigned_player

