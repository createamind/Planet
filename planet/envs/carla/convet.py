import cv2
import numpy as np
import os
import subprocess
from os.path import isfile, join
import time


def images_to_video():
    videos_dir = os.path.join('./', "Videos")
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
    ffmpeg_cmd = (
        "ffmpeg -loglevel -8 -r 60 -f image2 -s {x_res}x{y_res} "
        "-start_number 0 -i "
        "%08d.png -vcodec libx264 {vid}.mp4 && rm -f *.png "
    ).format(
        x_res=96,
        y_res=96,
        vid=os.path.join(videos_dir, str(time.time())))
    print("Executing ffmpeg command", ffmpeg_cmd)
    subprocess.call(ffmpeg_cmd, shell=True)


if __name__ == "__main__":
    images_to_video()