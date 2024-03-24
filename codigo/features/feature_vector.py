import math
from pathlib import Path
import librosa
import librosa.feature
import librosa.feature.rhythm
import numpy as np

import skimage

from . import lbp, lpq
from . import lbp, lpq

# rhythm patterns
# add cwd to module path
import os
import sys

sys.path.append(os.getcwd())
from vendor.rp_extract.rp_extract import rp_extract


def fv_lbp_rp_ex(audio_path: Path, spectrogram_path: Path):
    img = skimage.io.imread(spectrogram_path, as_gray=True)

    lbp_hist = lbp.get_lbp_histogram(img, p=8, r=2)

    # use rhythm pattern extraction from vendor/rp_extract
    x, fs = librosa.load(audio_path, sr=44100, mono=True)
    rp = rp_extract(x, fs, extract_rp=True)["rp"]

    return np.hstack((lbp_hist, rp))


def fv_lbp_ex(audio_path: Path, spectrogram_path: Path):
    img = skimage.io.imread(spectrogram_path, as_gray=True)

    lbp_hist = lbp.get_lbp_histogram(img, p=8, r=2)
    return np.hstack((lbp_hist,))


def fv_rp_ex(audio_path: Path, spectrogram_path: Path):
    # use rhythm pattern extraction from vendor/rp_extract
    x, fs = librosa.load(audio_path, sr=44100, mono=True)
    rp = rp_extract(x, fs, extract_rp=True)["rp"]
    return np.hstack((rp,))


def fv_lpq_ex(audio_path: Path, spectrogram_path: Path):
    img = skimage.io.imread(spectrogram_path, as_gray=True)

    lpq_hist = lpq.get_lpq_histogram(img)
    return np.hstack((lpq_hist,))


def fv_glcm_ex(audio_path: Path, spectrogram_path: Path):
    img = skimage.io.imread(spectrogram_path, as_gray=True)
    mats = skimage.feature.graycomatrix(
        img, [1, 2], [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4], levels=256
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    vals = sum(
        (skimage.feature.graycoprops(mats, prop).flatten().tolist() for prop in props),
        [],
    )
    return np.hstack((vals,))


def fv_mfcc_ex(audio_path: Path, spectrogram_path: Path):
    y, fs = librosa.load(audio_path, sr=44100, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13)

    mfccs = np.vstack(
        (
            np.mean(mfccs, axis=1),
            np.median(mfccs, axis=1),
            np.max(mfccs, axis=1),
            np.min(mfccs, axis=1),
            np.std(mfccs, axis=1),
        )
    )

    return np.hstack((mfccs.flatten(),))


def fv_glcm_mfcc_ex(audio_path: Path, spectrogram_path: Path):
    glcm = fv_glcm_ex(audio_path, spectrogram_path)
    mfccs = fv_mfcc_ex(audio_path, spectrogram_path)

    return np.hstack(
        (
            glcm,
            mfccs,
        )
    )


def fv_lbp_mfcc_ex(audio_path: Path, spectrogram_path: Path):
    lbp = fv_lbp_ex(audio_path, spectrogram_path)
    mfccs = fv_mfcc_ex(audio_path, spectrogram_path)

    return np.hstack(
        (
            lbp,
            mfccs,
        )
    )


def fv_lpq_mfcc_ex(audio_path: Path, spectrogram_path: Path):
    lpq = fv_lpq_ex(audio_path, spectrogram_path)
    mfccs = fv_mfcc_ex(audio_path, spectrogram_path)

    return np.hstack(
        (
            lpq,
            mfccs,
        )
    )


def fv_lbp_mfcc_glcm_ex(audio_path: Path, spectrogram_path: Path):
    lbp = fv_lbp_ex(audio_path, spectrogram_path)
    mfccs = fv_mfcc_ex(audio_path, spectrogram_path)
    glcm = fv_glcm_ex(audio_path, spectrogram_path)

    return np.hstack((lbp, mfccs, glcm))


fv_lbp_rp = ("lbp-rp", fv_lbp_rp_ex)
fv_lbp = ("lbp", fv_lbp_ex)
fv_rp = ("rp", fv_rp_ex)
fv_lpq = ("lpq", fv_lpq_ex)
fv_glcm = ("glcm", fv_glcm_ex)
fv_mfcc = ("mfcc", fv_mfcc_ex)
fv_glcm_mfcc = ("glcm-mfcc", fv_glcm_mfcc_ex)
fv_lbp_mfcc = ("lbp-mfcc", fv_lbp_mfcc_ex)
fv_lpq_mfcc = ("lpq-mfcc", fv_lpq_mfcc_ex)
fv_lbp_mfcc_glcm = ("lbp-mfcc-glcm", fv_lbp_mfcc_glcm_ex)
