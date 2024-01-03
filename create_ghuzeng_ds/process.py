import typing as T
import re
from pathlib import Path

from pydub import AudioSegment, silence
from pydub.effects import normalize
import numpy as np
from pydub.playback import play
import noisereduce as nr
import click 


def load(path: str) -> AudioSegment: 
    "loads audio as pydub AudioSegment" 
    clip = AudioSegment.from_file(path)
    clip = clip.set_channels(1)
    return clip 

def new_out_path(in_path: str, out_dir: str) -> str: 
    """
    makes a new output path for the processed file. Makes sure 
    the output subdirectory exits
    """
    in_path = Path(in_path)
    style = None
    try: 
        if 'Alienated' in str(in_path):
            style = 'Alienated'
        else: 
            style = re.search(r'_ ([A-Za-z]+) [Ss]chool', str(in_path.stem)).group(1).strip()
    except AttributeError: 
        style = 'unknown'
    new_path = Path(out_dir) / Path(style) / in_path.name
    new_path.parent.mkdir(exist_ok=True, parents=True)
    return new_path


def process_single_file(path: str, 
                        out_dir: str, 
                        noise_path: str, 
                        prop_decrease: float, 
                        silence_thresh_dbfs: int, 
                        keep_silence_ms: int): 
    clip = load(path) 
    assert clip.channels == 1, "Processing should be in mono"
    frame_rate = clip.frame_rate

    clip_norm = normalize(clip)
    clip_raw = np.array(clip_norm.get_array_of_samples())

    noise = load(noise_path)
    noise_raw = np.array(noise.get_array_of_samples())


    clip_raw_no_noise = nr.reduce_noise(y=clip_raw, 
                                        sr=frame_rate, 
                                        y_noise=noise_raw, 
                                        stationary=True, 
                                        prop_decrease=prop_decrease
                                        )
    clip_no_noise = AudioSegment(clip_raw_no_noise.tobytes(), 
                                 frame_rate=frame_rate, 
                                 channels=clip.channels, 
                                 sample_width=clip_raw_no_noise.dtype.itemsize)

    non_silent = silence.split_on_silence(clip_no_noise, 
                                          silence_thresh=silence_thresh_dbfs, 
                                          keep_silence=keep_silence_ms)
    clip_no_silence = sum(non_silent[1:], start=non_silent[0]) if len(non_silent) > 1 else non_silent[0]
    assert len(clip_no_silence) <= len(clip), "clip length has been increased!"

    print(f'Old duration: {len(clip)/1000:4.2f}s -> new duration: {len(clip_no_silence)/1000:4.2f}s')

    new_path = new_out_path(path, out_dir=out_dir)
    with Path(new_path).open('wb') as f: 
        clip_no_silence.export(f, format=new_path.suffix[1:])



@click.command()
@click.argument('input-dir')
@click.option('--extension', default='wav', show_default=True)
@click.option('--out-dir', default='./processed', show_default=True)
@click.option('--noise-path', default = 'noise_from_alienated 1 靖沐第一次錄音.wav', show_default=True)
@click.option('--prop-decrease', default = 0.8, show_default=True) 
@click.option('--silence-thresh-dbfs', default = -40, show_default=True) 
@click.option('--keep-silence-ms', default = 500, show_default=True) 
def process(input_dir: str, 
            extension: str, 
            **kwargs):

    input_dir = Path(input_dir).resolve()
    glob_str = f'*.{extension}'

    input_paths = list(input_dir.rglob(glob_str))
    print(f'Processing {len(input_paths)} files')
    for input_path in input_paths: 
        print(f'Processing {input_path.name}')
        process_single_file(input_path, **kwargs)


if __name__ == '__main__':
    process() 