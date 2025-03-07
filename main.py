# import cv2
import numpy as np
from src.utils import VideoUtils
from src.tracker import Tracker
from src.team_assigner import TeamAssigner
from src.ball_assigner import BallAssigner
from src.camera_movement_estimator import CameraMovementEstimator
from src.view_transformer import ViewTransformer

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    print("Initializing video utilities...\n")
    vu = VideoUtils()

    print("Initializing tracker with model...\n")
    tracker = Tracker('./models/best.pt')

    # Input and output paths
    input_video_path = './input_videos/input.mp4'
    output_video_path = './output_videos/output.mp4'
    tracks_stub_path = './stubs/track_stubs.pkl'
    camera_movement_stub_path = './stubs/camera_movement_stubs.pkl'

    print(f"Reading video frames from: {input_video_path}")
    video_frames = vu.read_video(input_video_path)
    print(f"Total frames loaded: {len(video_frames)}\n")

    print("Getting object tracks...")
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True,
                                       stub_path=tracks_stub_path)
    print("Object tracking complete.\n")

    print("Adding position to tracks...")
    tracker.add_position_to_tracks(tracks)
    print("Adding position to tracks complete.\n")

    print("Estimating camera movements...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, 
                                                                              read_from_stub=True,
                                                                              stub_path=camera_movement_stub_path)
    print("Camera movement estimation complete.\n")

    print("Adjusting track positions...")
    camera_movement_estimator.adjust_track_positions(tracks, camera_movement_per_frame)
    print("Adjusting track positions complete.\n")

    print("Adjusting view transformations...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    print("Adjusting view transformations complete.\n")

    print("Interpolating ball positions...")
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])
    print("Ball position interpolation complete.\n")

    print("Initializing team assigner...")
    team_assigner = TeamAssigner()
    print("Assigning team colors for the first frame...")
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    print("Team color assignment complete.\n")

    print("Assigning teams to players across frames...")
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    print("Player team assignment complete.\n")

    print("Initializing ball assigner...")
    ball_assigner = BallAssigner()
    team_ball_control = []

    print("Assigning ball possession...")
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']  # track ID 1 for ball
        assigned_player = ball_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)  # handle initial case
    print("Ball possession assignment complete.\n")

    team_ball_control = np.array(team_ball_control)

    print("Drawing object tracks on frames...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    print("Object tracks drawn.\n")

    print("Drawing camera movement indicators...")
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    print("Camera movement drawing complete.\n")

    print(f"Saving output video to: {output_video_path}")
    vu.save_video(output_video_frames, output_video_path)
    print("Video saved successfully.\n")

if __name__ == '__main__':
    main()
