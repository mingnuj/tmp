import numpy as np
import os

def landmark_reader(npy_path):
    """
    npy_path: outputs/landmarks/<video_name>
    """
    npy_info = []
    video_name = npy_path.split('/')[-1]
    for frames in os.listdir(npy_path):
        for ids in os.listdir(os.path.join(npy_path, frames)):
            if ids.endswith("npy"):
                npy_info.append({"VideoName":video_name, "frame_number":int(frames),
                                 "ID":int(ids[:-4]), "landmarks":np.load(os.path.join(npy_path, frames, ids))})
    return npy_info

file_path = "output/landmarks/recording"

example = landmark_reader(file_path)

# Output example
for info in example:
    print("video name: {}\nframe number: {}\nface id: {}, landmark shape: {}\n\n".format(
        info["VideoName"], info["frame_number"], info["ID"], info["landmarks"].shape))

