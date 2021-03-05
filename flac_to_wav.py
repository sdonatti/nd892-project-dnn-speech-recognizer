import os
import pathlib
import sys

for file in pathlib.Path(sys.argv[1]).rglob('*.flac'):
    os.system(f'ffmpeg -y -f flac -i {str(file)} -ab 64k -ac 1 -ar 16000 -f wav {str(file)[:-5]}.wav')
