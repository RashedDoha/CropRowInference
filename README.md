# How to make inferences on an input video
---
1. Clone this Repo
2. Download input video and pretrained model file from shared Box folder.
3. Place input video and pretrained model files in root directory of cloned repo.
4. cd into the root directory of cloned repo.
5. Run the script `generate_video.py` using the command line arguments `--video_src` and `--video_dst`.

---
Example test command

`python generate_vide.py --video_src='crop-row.mp4' --video_dst='crop-row-preds.mp4'`

Other command line arguments can be provided. Use the `-help` flag for more options.