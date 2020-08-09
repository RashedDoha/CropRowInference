import argparse

parser = argparse.ArgumentParser(description='Video Processing Arguements.')
parser.add_argument('--video_src', type=str, help='Path to the input video file.')
parser.add_argument('--video_dst', type=str, help='Path to the destination video file.')
parser.add_argument('--video_fps', type=int, default=25, help='FPS of video output.')
parser.add_argument('--model_src', type=str, default='model.pth', help='Name of the model file.')
parser.add_argument('--model_arch', type=str, default='densenet', help='Model Architecture to use.')
parser.add_argument('--num_frames', type=int, default=10000, help='Number of frames to process.')
