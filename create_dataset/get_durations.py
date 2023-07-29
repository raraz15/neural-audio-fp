import os
import json

import essentia.standard as es

my_dir = "/home/oaraz/data/discotube"
discotube_dir = "/mnt/projects/discotube/discotube-2023-03/audio-new/audio"
audio_dir = my_dir+discotube_dir

txt_dir = "/home/oaraz/nextcore/afp/discotube_subset"
name = "discotube-2023.txt.filtered.shuf.188000"

if __name__=="__main__":

    with open(os.path.join(txt_dir, name)) as f:
        mp4_paths = [line.strip() for line in f.readlines()]

    _mp4_paths = []
    for mp4_path in mp4_paths:
        if os.path.exists(my_dir+mp4_path):
            _mp4_paths.append(my_dir+mp4_path)
    mp4_paths = _mp4_paths

    for i,audio_path in enumerate(mp4_paths):
        if (i+1) % 10000 == 0:
            print(i+1)
        audio, fs, _, _, _, _ = es.AudioLoader(filename=audio_path)()
        T = len(audio) / fs
        with open(f'{name}.durations', 'a') as f:
            f.write(json.dumps({audio_path.split("/")[-1]: T})+"\n")