# import cv2
import numpy as np
from src.utils import VideoUtils
from src.tracker import Tracker
from src.team_assigner import TeamAssigner
from src.ball_assigner import BallAssigner

def main():
    # initialize video utils
    vu = VideoUtils()

    # initialize tracker
    tracker = Tracker('./models/best.pt')

    # input and output paths
    input_video_path = './input_videos/input.mp4'
    output_video_path = './output_videos/output.mp4'
    stub_path = './stubs/track_stubs.pkl'

    # get input video frames
    video_frames =  vu.read_video(input_video_path)

    # get tracks of video frames
    tracks =  tracker.get_object_tracks(video_frames, 
                              read_from_stub=True,
                              stub_path=stub_path)
    
    # interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

    #crop player image
    # for track_id, player in tracks['players'][0].items():
    #     x1, y1, x2, y2 = player['bbox']
    #     frame = video_frames[0]
        
    #     # get cropped image
    #     cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]

    #     # save cropped image
    #     cv2.imwrite(f'./output_images/cropped_image.jpg', cropped_image)

    #     break

    # assign player teams only in first frame
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # assign ball to player
    ball_assigner = BallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        # 1 is track id for ball
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = ball_assigner.assign_ball_to_player(player_track, ball_bbox) 

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # assign ball control to current team
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # assign ball control to last team which was controlling ball
            team_ball_control.append(team_ball_control[-1])
            
    team_ball_control = np.array(team_ball_control)

    # draw output
    ## draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # save video
    vu.save_video(output_video_frames, output_video_path)

if __name__ == '__main__':
    main()