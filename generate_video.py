from config import parser
from utils import get_video, get_model, write_video, view_predicted_annotation
import torch
import cv2

if __name__ == '__main__':
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    video = get_video(args.video_src)
    video.set(cv2.CAP_PROP_FPS, args.video_fps)
    model = get_model(args.model_arch, args.model_src, device=device)
    model.eval()
    frames = []
    coefs = None
    print(f'Processing Video for #{args.num_frames} frames.')
    for i in range(args.num_frames):
        success, image = video.read()
        try:
            image = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))
            diff, coefs, t_image = view_predicted_annotation(model,image,device)
            frames.append(image)
        except:
            pass

        

    print(f'Writing Video as {args.video_dst}...')
    print(len(frames))
    write_video(frames, args.video_fps, args.video_dst)
