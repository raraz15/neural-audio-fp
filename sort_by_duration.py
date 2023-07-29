import os

import essentia.standard as es

fs=8000
N_segments = 61
T_segment = (1 + (2*0.5))
L0 = int(T_segment*fs)

L_min = L0*(1+(N_segments-1)/2)
T_min = L_min/fs
print('T_min = ', T_min)

my_dir = "/home/oaraz/data/discotube"
discotube_dir = "/mnt/projects/discotube/discotube-2023-03/audio-new/audio"
audio_dir = my_dir+discotube_dir

txt_dir = "/home/oaraz/nextcore/afp/discotube_subset"
name = "discotube-2023.txt.filtered.shuf.188000"

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
    audio = es.MonoLoader(filename=audio_path, 
                        sampleRate=44100)()
    T = len(audio) / 44100
    if T < T_min:
        with open('short_audio_paths.txt', 'a') as f:
            f.write(audio_path + '\n')
    else:
        with open('long_audio_paths.txt', 'a') as f:
            f.write(audio_path + '\n')