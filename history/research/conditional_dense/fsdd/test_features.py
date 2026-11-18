"""Check the frozen MFCC layout and pre-emphasis on controlled synthetic audio."""
import numpy as np
from research.conditional_dense.fsdd.features import mfcc


def test_silent_single_frame():
    output = mfcc(np.zeros(17,dtype=np.int16),8000).reshape(16,13)
    expected = np.zeros((16,13))
    expected[:,0] = np.sqrt(26)*np.log(1e-12)
    np.testing.assert_allclose(output,expected,atol=1e-5)


def test_explicit_filterbank_and_cosine_transform():
    rate = 8000
    samples = np.random.default_rng(7827).integers(-32000,32000,461,dtype=np.int16)
    signal = samples.astype(float)/32768
    pre = np.r_[signal[0],signal[1:]-.97*signal[:-1]]
    cuts = 700*np.expm1(np.linspace(0,np.log1p((rate/2)/700),28))
    frequencies = np.arange(129)*rate/256
    cosines = np.cos(np.pi*np.outer(np.arange(13),np.arange(26)+.5)/26)*np.sqrt(2/26)
    cosines[0] /= np.sqrt(2)
    frames = []
    for offset in range(0,321,80):
        end = min(len(pre),offset+200)
        frame = np.zeros(200)
        frame[:end-offset] = pre[offset:end]
        powers = abs(np.fft.rfft(frame*np.hamming(200),256))**2/256
        values = []
        for i in range(26):
            left,peak,right = cuts[i:i+3]
            weighted = [p*max(0,min((f-left)/(peak-left),(right-f)/(right-peak))) for f,p in zip(frequencies,powers)]
            values.append(np.log(max(1e-12,sum(weighted))))
        frames.append(cosines @ values)
    stacked = np.array(frames)
    oracle = np.stack([np.interp(np.linspace(0,4,16),np.arange(5),stacked[:,j]) for j in range(13)],axis=1)
    np.testing.assert_allclose(mfcc(samples,rate).reshape(16,13),oracle,rtol=1e-6,atol=1e-5)
