import os
import shutil
from pathlib import Path
import subprocess as sp


class WavConverter(object):
    def __init__(self, source_path: str, dest_path: str, samplerate: int=16000, mode="wav"):
        assert mode in ["wav", "mp3"]
        self.source_path = Path(source_path)
        self.dest_path = Path(dest_path)
        self.ffmpeg_bin = "ffmpeg"
        self.samplerate = samplerate
        self.mode = mode

    def convert_audio(self, source, destination):
        command = [self.ffmpeg_bin,
                   '-loglevel', 'error',
                   '-i', source,
                   '-ac', '1',
                #    '-acodec', 'adpcm_ms',
                   '-ar', '{}'.format(self.samplerate),
                   '{}'.format(destination)]
        sp.run(command)

    def convert_to_mp3(self, source, destination):
        command = [self.ffmpeg_bin,
                   '-loglevel', 'error',
                   '-i', source,
                   '-ab', '160k',
                   '-map_metadata', '0',
                   '-id3v2_version', '3',
                   '{}'.format(destination)]
        sp.run(command)

    def copy_content(self, exclude_pattern="**/*.wav"):
        files_to_copy = list(set(self.source_path.glob("**/*")) - set(self.source_path.glob(exclude_pattern)))
        copied_objects = 0
        folders_objects = 0
        for file in tqdm(files_to_copy, desc="copy files"):
            if file.is_dir():
                folders_objects += 1
                continue
            dest_file = self.dest_path.joinpath(Path(file).relative_to(self.source_path))
            shutil.copy(file, dest_file)
            copied_objects += 1
        print("{} objects was successful copied".format(copied_objects))
        print("{} folders skipped".format(folders_objects))

    def __call__(self, file):
        if self.mode == "wav":
            dest_path = self.dest_path.joinpath(Path(file).relative_to(self.source_path)).with_suffix(".wav")
        else:
            dest_path = self.dest_path.joinpath(Path(file).relative_to(self.source_path)).with_suffix(".mp3")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            return None

        self.convert_audio(file, dest_path)


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='sound resampler')

    parser.add_argument('--source_folder',
                        type=str,
                        default=False,
                        required=True,
                        help="path dataset with input wav")
    parser.add_argument('--dest_folder',
                        type=str,
                        default=False,
                        required=True,
                        help="path dataset with output wav")
    parser.add_argument('--samplerate',
                        type=int,
                        default=16000,
                        required=False,
                        help="target samplerate")
    parser.add_argument('--job',
                        type=int,
                        default=1,
                        required=False,
                        help="num jobs")

    parsed = parser.parse_args()

    source_folder = parsed.source_folder
    dest_folder = parsed.dest_folder
    samplerate = parsed.samplerate
    njobs = parsed.job

    if os.path.exists(os.path.join(dest_folder, '.resampled')):
        print("{} already exists and resampled".format(dest_folder))
        exit()

    files = list(Path(source_folder).glob("**/*.wav"))
    converter = WavConverter(source_path=source_folder,
                             dest_path=dest_folder,
                             mode="wav",
                             samplerate=samplerate)

    # list(map(converter, files))
    p = mp.Pool(njobs)
    for _ in tqdm(p.imap_unordered(converter, files), total=len(files), smoothing=0.1):
        pass
    converter.copy_content(exclude_pattern="**/*.wav")

    open(os.path.join(dest_folder, '.resampled'), 'a').close()
