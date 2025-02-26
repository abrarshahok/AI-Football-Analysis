from src.utils import VideoUtils
from src.trackers import Tracker

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

    # draw output
    ## draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    vu.save_video(output_video_frames, output_video_path)

if __name__ == '__main__':
    main()