# translation of lpq.m from matlab to python, using numpy et al

import enum
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from scipy.linalg import svd


class FrequencyEstimationMethod(enum.StrEnum):
    STFT_UNIFORM_WINDOW = "STFT with uniform window"
    STFT_GAUSSIAN_WINDOW = "STFT with Gaussian window"
    GAUSSIAN_DERIVATIVE_QUADRATURE_FILTER_PAIR = (
        "Gaussian derivative quadrature filter pair"
    )


class OutputFormat(enum.StrEnum):
    NORMALIZED_HISTOGRAM = "nh"
    HISTOGRAM = "h"
    IMAGE = "im"


def lpq(
    img,
    winSize: int = 7,
    decorr: bool = True,
    freqestim: FrequencyEstimationMethod = FrequencyEstimationMethod.STFT_UNIFORM_WINDOW,
    mode: OutputFormat = OutputFormat.NORMALIZED_HISTOGRAM,
):
    """
    Computes the Local Phase Quantization (LPQ) descriptor
    for the input image img. Descriptors are calculated
    using only valid pixels i.e. size(img)-(winSize-1).

    ## Parameters

    img : ndarray
        Input image (2D grayscale).
    winSize : int, optional
        Size of the local window. winSize must be odd number and greater or equal to 3.
    decorr : boolean, optional
        True to perform decorrelation step (decorrelation is done by default).
    freqestim : int, optional
        indicates which method is used for local frequency estimation. Possible values are:
            1 -> STFT with uniform window (corresponds to basic version of LPQ)
            2 -> STFT with Gaussian window (equals also to Gaussian quadrature filter pair)
            3 -> Gaussian derivative quadrature filter pair.
    mode : str, optional
        defines the desired output type. Possible choices are:
            'nh' -> normalized histogram of LPQ codewords (1*256 double vector, for which sum(result)==1)
            'h'  -> un-normalized histogram of LPQ codewords (1*256 double vector)
            'im' -> LPQ codeword image ([size(img,1)-r,size(img,2)-r] double matrix)

    ## Output
    1*256 double or size(img)-(winSize-1) uint8, LPQ descriptors histogram or LPQ code image (see "mode" above)
    """

    rho = 0.90

    STFTalpha = 1 / winSize
    sigmaS = (winSize - 1) / 4
    sigmaA = 8 / (winSize - 1)

    convmode = "valid"

    if img.ndim != 2:
        raise ValueError("Input image must be 2D grayscale")
    if winSize < 3 or winSize % 2 == 0:
        raise ValueError("Window size winSize must be odd number >= 3")

    img = img.astype(np.float64)
    r = (winSize - 1) / 2
    x = np.arange(-r, r + 1)
    u = np.arange(1, r + 1)

    if (
        freqestim == FrequencyEstimationMethod.STFT_UNIFORM_WINDOW
    ):  # STFT uniform window
        # Basic STFT filters
        w0 = x * 0 + 1
        w1 = np.exp(1j * -2 * np.pi * x * STFTalpha)
        w2 = np.conj(w1)
    elif (
        freqestim == FrequencyEstimationMethod.STFT_GAUSSIAN_WINDOW
    ):  # STFT Gaussian window
        # Basic STFT filters
        w0 = x * 0 + 1
        w1 = np.exp(1j * -2 * np.pi * x * STFTalpha)
        w2 = np.conj(w1)

        # Gaussian window
        gs = np.exp(-0.5 * (x / sigmaS) ** 2) / (np.sqrt(2 * np.pi) * sigmaS)

        # Windowed filters
        w0 = gs * w0
        w1 = gs * w1
        w2 = gs * w2

        # Normalize to zero mean
        w1 = w1 - np.mean(w1)
        w2 = w2 - np.mean(w2)
    elif (
        freqestim
        == FrequencyEstimationMethod.GAUSSIAN_DERIVATIVE_QUADRATURE_FILTER_PAIR
    ):  # Gaussian derivative quadrature filters
        # Frequency domain definition of filters
        G0 = np.exp(-(x**2) * ((np.sqrt(2) * sigmaA) ** 2))
        G1 = np.concatenate(
            np.zeros((1, len(u))), 0, u * np.exp(-(u**2) * sigmaA**2)
        )

        # Normalize to avoid small numerical values (do not change the phase response we use)
        G0 = G0 / np.max(np.abs(G0))
        G1 = G1 / np.max(np.abs(G1))

        # Compute spatial domain correspondences of the filters
        w0 = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(G0))))
        w1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(G1)))
        w2 = np.conj(w1)

        # Normalize to avoid small numerical values (do not change the phase response we use)
        w0 = w0 / np.max(np.abs([np.real(np.max(w0)), np.imag(np.max(w0))]))
        w1 = w1 / np.max(np.abs([np.real(np.max(w1)), np.imag(np.max(w1))]))
        w2 = w2 / np.max(np.abs([np.real(np.max(w2)), np.imag(np.max(w2))]))

    ## Run filters to compute the frequency response in the four points. Store real and imaginary parts separately
    # Run first filter
    filterResp = convolve2d(
        convolve2d(img, np.transpose(w0)[:, None], mode="valid"),
        w1[None, :],
        mode="valid",
    )

    # Initilize frequency domain matrix for four frequency coordinates (real and imaginary parts for each frequency)
    freqResp = np.zeros((filterResp.shape[0], filterResp.shape[1], 8), dtype=complex)
    # Store filter outputs
    freqResp[:, :, 0] = np.real(filterResp)
    freqResp[:, :, 1] = np.imag(filterResp)
    # Repeat the procedure for other frequencies
    filterResp = convolve2d(
        convolve2d(img, np.transpose(w1)[:, None], mode="valid"),
        w0[None, :],
        mode="valid",
    )
    freqResp[:, :, 2] = np.real(filterResp)
    freqResp[:, :, 3] = np.imag(filterResp)

    filterResp = convolve2d(
        convolve2d(img, np.transpose(w1)[:, None], mode="valid"),
        w1[None, :],
        mode="valid",
    )

    freqResp[:, :, 4] = np.real(filterResp)
    freqResp[:, :, 5] = np.imag(filterResp)

    filterResp = convolve2d(
        convolve2d(img, np.transpose(w1)[:, None], mode="valid"),
        w2[None, :],
        mode="valid",
    )

    freqResp[:, :, 6] = np.real(filterResp)
    freqResp[:, :, 7] = np.imag(filterResp)

    # Read the size of frequency matrix
    (freqRow, freqCol, freqNum) = freqResp.shape

    ## If decorrelation is used, compute covariance matrix and corresponding whitening transform
    if decorr:
        # Compute covariance matrix (covariance between pixel positions x_i and x_j is rho^||x_i-x_j||)
        xp, yp = np.meshgrid(np.arange(1, winSize + 1), np.arange(1, winSize + 1))
        pp = np.column_stack((xp.flatten(), yp.flatten()))
        dd = cdist(pp, pp, metric="euclidean")
        C = rho**dd

        # Form 2-D filters q1, q2, q3, q4 and corresponding 2-D matrix operator M (separating real and imaginary parts)
        q1 = np.outer(np.transpose(w0), w1)
        q2 = np.outer(np.transpose(w1), w0)
        q3 = np.outer(np.transpose(w1), w1)
        q4 = np.outer(np.transpose(w1), w2)
        u1, u2 = np.real(q1), np.imag(q1)
        u3, u4 = np.real(q2), np.imag(q2)
        u5, u6 = np.real(q3), np.imag(q3)
        u7, u8 = np.real(q4), np.imag(q4)
        M = np.vstack(
            (
                u1.flatten(),
                u2.flatten(),
                u3.flatten(),
                u4.flatten(),
                u5.flatten(),
                u6.flatten(),
                u7.flatten(),
                u8.flatten(),
            )
        )

        # Compute whitening transformation matrix V
        D = M @ C @ M.T
        A = np.diag(
            [1.000007, 1.000006, 1.000005, 1.000004, 1.000003, 1.000002, 1.000001, 1]
        )
        U, S, V = svd(A @ D @ A)

        # Reshape frequency response
        freqResp = freqResp.reshape((freqRow * freqCol, freqNum))

        # Perform whitening transform
        freqResp = np.dot(V.T, freqResp.T).T

        # Undo reshape
        freqResp = freqResp.reshape((freqRow, freqCol, freqNum))

    ## Perform quantization and compute LPQ codewords
    LPQdesc = np.zeros((freqRow, freqCol))
    for i in range(freqNum):
        LPQdesc = LPQdesc + (np.double(np.real(freqResp[:, :, i])) > 0) * (2**i)

    ## Switch format to uint8 if LPQ code image is required as output
    if mode == OutputFormat.IMAGE:
        LPQdesc = LPQdesc.astype(np.uint8)

    ## Histogram if needed
    if mode == OutputFormat.HISTOGRAM or mode == OutputFormat.NORMALIZED_HISTOGRAM:
        LPQdesc = np.histogram(LPQdesc.flatten(), bins=np.arange(257))[0]

    ## Normalize histogram if needed
    if mode == OutputFormat.NORMALIZED_HISTOGRAM:
        LPQdesc = LPQdesc / np.sum(LPQdesc)

    return LPQdesc


def get_lpq_histogram(im):
    hist = lpq(im, winSize=7, mode=OutputFormat.NORMALIZED_HISTOGRAM)
    return hist
