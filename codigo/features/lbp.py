import skimage
import numpy as np
import math


def rol(n, rotations, width=8):
    return ((n << rotations) & ((1 << (width - 1)) - 1)) | (n >> (width - rotations))


def get_lbp_histogram(im, p, r):
    # calculate uniform lbp for image
    lbp = skimage.feature.local_binary_pattern(im, p, r, method="nri_uniform")

    # store number of occurences in histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    return hist


def get_rlbp_histogram(im, p, r):
    # calculate regular lbp for image
    lbp = skimage.feature.local_binary_pattern(im, p, r, method="default")

    # store number of occurences in histogram
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, density=False, bins=n_bins, range=(0, n_bins))

    rlbp_hist = np.zeros((n_bins,), lbp_hist.dtype)

    # 11010000

    # for all bit patterns k of length p:
    # find all occurences of bit patterns 101 and 010 in k
    # print all occurences
    for k in range(n_bins):
        # check if k is non-uniform
        # k is non-uniform if it has more than 2 bit flips

        flips = k ^ rol(k, p - 1, p)
        cnt = flips.bit_count()

        if cnt > 2:
            different = False
            for rot in range(p):
                krot = rol(k, rot, p)
                if (krot & 0b111) == 0b101:
                    rrot = krot | 0b111
                    tgtk = rol(rrot, p - rot, p)
                    rlbp_hist[tgtk] += lbp_hist[k]
                    different = True
                elif (krot & 0b111) == 0b010:
                    rrot = krot ^ (krot & 0b111)
                    tgtk = rol(rrot, p - rot, p)
                    rlbp_hist[tgtk] += lbp_hist[k]
                    different = True
            if not different:
                rlbp_hist[k] += lbp_hist[k]
        else:
            rlbp_hist[k] += lbp_hist[k]

    # finally, restrict histogram to only uniform patterns
    # group non-uniform patterns in extra bucket
    # there are C(p, 2) * 2 + 2 uniform patterns + 1 non-uniform bucket
    final_hist = np.zeros((math.comb(p, 2) * 2 + 2 + 1,), lbp_hist.dtype)

    occurences = sum(rlbp_hist)
    final_hist[0] = rlbp_hist[0]
    final_hist[1] = rlbp_hist[2**p - 1]
    i = 2

    ALL_ONES = 2**p - 1
    for l in range(1, p):
        num = (ALL_ONES ^ (ALL_ONES << l)) & ALL_ONES
        for k in range(p):
            final_hist[i] = rlbp_hist[num]
            i += 1
            num = rol(num, 1, p)

    final_hist[i] = sum(final_hist[0:i]) - occurences
    final_hist = final_hist.astype(np.float32) / sum(final_hist)

    return final_hist
