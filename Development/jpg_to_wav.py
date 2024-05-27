#!/usr/bin/env python
# coding: utf-8

# In[50]:


import torch
import torchaudio
import numpy as np
from PIL import Image
import typing as T
from dataclasses import dataclass
from enum import Enum
import io
import numpy as np
import pydub
from scipy.io import wavfile
from diffusers import StableDiffusionPipeline


device = 'cuda'



def spectrogram_from_image(
    image: Image.Image,
    power: float = 0.25,
    stereo: bool = False,
    max_value: float = 30e6,
) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    This is the inverse of image_from_spectrogram, except for discretization error from
    quantizing to uint8.

    Args:
        image: (frequency, time, channels)
        power: The power curve applied to the spectrogram
        stereo: Whether the spectrogram encodes stereo data
        max_value: The max value of the original spectrogram. In practice doesn't matter.

    Returns:
        spectrogram: (channels, frequency, time)
    """
    # Convert to RGB if single channel
    if image.mode in ("P", "L"):
        image = image.convert("RGB")

    # Flip Y
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Munge channels into a numpy array of (channels, frequency, time)
    data = np.array(image).transpose(2, 0, 1)
    if stereo:
        # Take the G and B channels as done in image_from_spectrogram
        data = data[[1, 2], :, :]
    else:
        data = data[0:1, :, :]

    # Convert to floats
    data = data.astype(np.float32)

    # Invert
    data = 255 - data

    # Rescale to 0-1
    data = data / 255

    # Reverse the power curve
    data = np.power(data, 1 / power)

    # Rescale to max value
    data = data * max_value

    return data


def audio_from_waveform(
    samples: np.ndarray, sample_rate: int, normalize: bool = False
) -> pydub.AudioSegment:
    """
    Convert a numpy array of samples of a waveform to an audio segment.

    Args:
        samples: (channels, samples) array
    """
    # Normalize volume to fit in int16
    if normalize:
        samples *= np.iinfo(np.int16).max / np.max(np.abs(samples))

    # Transpose and convert to int16
    samples = samples.transpose(1, 0)
    samples = samples.astype(np.int16)

    # Write to the bytes of a WAV file
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples)
    wav_bytes.seek(0)

    # Read into pydub
    return pydub.AudioSegment.from_wav(wav_bytes)

@dataclass(frozen=True)
class SpectrogramParams:
    """
    Parameters for the conversion from audio to spectrograms to images and back.

    Includes helpers to convert to and from EXIF tags, allowing these parameters to be stored
    within spectrogram images.

    To understand what these parameters do and to customize them, read `spectrogram_converter.py`
    and the linked torchaudio documentation.
    """

    # Whether the audio is stereo or mono
    stereo: bool = False

    # FFT parameters
    sample_rate: int = 44100
    step_size_ms: int = 10
    window_duration_ms: int = 100
    padded_duration_ms: int = 400

    # Mel scale parameters
    num_frequencies: int = 512
    # TODO(hayk): Set these to [20, 20000] for newer models
    min_frequency: int = 0
    max_frequency: int = 10000
    mel_scale_norm: T.Optional[str] = None
    mel_scale_type: str = "htk"
    max_mel_iters: int = 200

    # Griffin Lim parameters
    num_griffin_lim_iters: int = 32

    # Image parameterization
    power_for_image: float = 0.25

    class ExifTags(Enum):
        """
        Custom EXIF tags for the spectrogram image.
        """

        SAMPLE_RATE = 11000
        STEREO = 11005
        STEP_SIZE_MS = 11010
        WINDOW_DURATION_MS = 11020
        PADDED_DURATION_MS = 11030

        NUM_FREQUENCIES = 11040
        MIN_FREQUENCY = 11050
        MAX_FREQUENCY = 11060

        POWER_FOR_IMAGE = 11070
        MAX_VALUE = 11080

    @property
    def n_fft(self) -> int:
        """
        The number of samples in each STFT window, with padding.
        """
        return int(self.padded_duration_ms / 1000.0 * self.sample_rate)

    @property
    def win_length(self) -> int:
        """
        The number of samples in each STFT window.
        """
        return int(self.window_duration_ms / 1000.0 * self.sample_rate)

    @property
    def hop_length(self) -> int:
        """
        The number of samples between each STFT window.
        """
        return int(self.step_size_ms / 1000.0 * self.sample_rate)

    def to_exif(self) -> T.Dict[int, T.Any]:
        """
        Return a dictionary of EXIF tags for the current values.
        """
        return {
            self.ExifTags.SAMPLE_RATE.value: self.sample_rate,
            self.ExifTags.STEREO.value: self.stereo,
            self.ExifTags.STEP_SIZE_MS.value: self.step_size_ms,
            self.ExifTags.WINDOW_DURATION_MS.value: self.window_duration_ms,
            self.ExifTags.PADDED_DURATION_MS.value: self.padded_duration_ms,
            self.ExifTags.NUM_FREQUENCIES.value: self.num_frequencies,
            self.ExifTags.MIN_FREQUENCY.value: self.min_frequency,
            self.ExifTags.MAX_FREQUENCY.value: self.max_frequency,
            self.ExifTags.POWER_FOR_IMAGE.value: float(self.power_for_image),
        }

    @classmethod
    def from_exif(cls, exif: T.Mapping[int, T.Any]): 
        """
        Create a SpectrogramParams object from the EXIF tags of the given image.
        """
        # TODO(hayk): Handle missing tags
        return cls(
            sample_rate=exif[cls.ExifTags.SAMPLE_RATE.value],
            stereo=bool(exif[cls.ExifTags.STEREO.value]),
            step_size_ms=exif[cls.ExifTags.STEP_SIZE_MS.value],
            window_duration_ms=exif[cls.ExifTags.WINDOW_DURATION_MS.value],
            padded_duration_ms=exif[cls.ExifTags.PADDED_DURATION_MS.value],
            num_frequencies=exif[cls.ExifTags.NUM_FREQUENCIES.value],
            min_frequency=exif[cls.ExifTags.MIN_FREQUENCY.value],
            max_frequency=exif[cls.ExifTags.MAX_FREQUENCY.value],
            power_for_image=exif[cls.ExifTags.POWER_FOR_IMAGE.value],
        )


params = SpectrogramParams()

inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
            n_fft=params.n_fft,
            n_iter=params.num_griffin_lim_iters,
            win_length=params.win_length,
            hop_length=params.hop_length,
            window_fn=torch.hann_window,
            power=1.0,
            wkwargs=None,
            momentum=0.99,
            length=None,
            rand_init=True,
        ).to(device)


inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_stft=params.n_fft // 2 + 1,
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(device)

def image_to_audio(img):
    spectrogram = spectrogram_from_image(img)
    amplitudes_mel = torch.from_numpy(spectrogram).to(device)
    # Reconstruct the waveform
    amplitudes_linear = inverse_mel_scaler(amplitudes_mel)
    waveform = inverse_spectrogram_func(amplitudes_linear)

    # Convert to audio segment
    segment = audio_from_waveform(
        samples=waveform.cpu().numpy(),
        sample_rate=params.sample_rate,
        # Normalize the waveform to the range [-1, 1]
        normalize=True,
    )
    return segment



# In[61]:


# pipeline = StableDiffusionPipeline.from_pretrained(
#             'riffusion-guzheng-v2')
# pipeline.to('cuda')
# generator = torch.Generator(device='cuda').manual_seed(42)


# # In[66]:


# prompts = ['lofi funk', 'happy pop', 'solo guzheng']
# for prompt in prompts:
#     images = pipeline('lofi funk', num_inference_steps=20, generator=generator)
#     audio = image_to_audio(images[0][0])
#     fname = f"riffusion-guzheng-v2/{prompt.replace(' ', '_')}.wav"
#     audio.export(fname)

