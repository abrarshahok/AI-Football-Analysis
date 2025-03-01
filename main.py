# import cv2
from src.utils import VideoUtils
from src.tracker import Tracker
from src.team_assigner import TeamAssigner

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



    # draw output
    ## draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    vu.save_video(output_video_frames, output_video_path)

if __name__ == '__main__':
    main()