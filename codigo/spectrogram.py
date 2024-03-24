from PIL import Image
import numpy as np
import librosa
import skimage
import matplotlib.pyplot as plt


def scale_to_range(x, min, max):
    x_std = (x - x.min()) / (x.max() - x.min())
    x_scaled = x_std * (max - min) + min
    return x_scaled


def generate_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=48000, mono=True)  # dataset está em 48kHz mono

    S = librosa.stft(y, n_fft=1024, hop_length=552, center=True)
    S = librosa.power_to_db(np.abs(S) ** 2, ref=np.max)

    return S


def save_spectrogram_image_to_path(spectrogram, path):
    img = scale_to_range(spectrogram, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)

    im = Image.fromarray(img)

    # map using plt colormap
    im.putpalette(
        (plt.get_cmap("viridis")(np.arange(256)) * 256).astype(np.uint8), rawmode="RGBA"
    )

    # convert im to BGR2GRAY
    im = im.convert("L")
    im.save(path)


def load_spectrogram_image_from_path(path):
    im = skimage.io.imread(path)
    return im
