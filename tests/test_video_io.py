import cv2
import numpy as np

from pipelines.data.video_io import read_video_rgb


def test_real_mp4_roundtrip(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16, (96, 96), False)
    assert writer.isOpened()
    for index in range(32):
        writer.write(np.full((96, 96), index * 7, dtype=np.uint8))
    writer.release()
    frames = read_video_rgb(path)
    assert frames.shape == (32, 96, 96, 3)
    assert frames.dtype == np.uint8
    assert frames[-1].mean() > frames[0].mean()
