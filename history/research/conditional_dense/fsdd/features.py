"""Per-utterance fixed MFCC features for the preregistered unseen-speaker study."""
import hashlib
import io
import re
from functools import lru_cache
from itertools import combinations
from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile

from research.conditional_dense.confirmation import Panel, SEEDS


@lru_cache(maxsize=None)
def filters(rate, nfft):
    nyquist = rate / 2
    top = 2595 * np.log10(1 + nyquist/700)
    points = 700 * (10**(np.linspace(0,top,28)/2595) - 1)
    freqs = np.fft.rfftfreq(nfft,1/rate)
    left, mid, right = points[:-2,None], points[1:-1,None], points[2:,None]
    return np.maximum(0.,np.minimum((freqs-left)/(mid-left), (right-freqs)/(right-mid)))


def mfcc(samples, rate):
    if samples.ndim != 1 or len(samples)==0 or rate <= 0 or samples.dtype.kind not in "iuf":
        raise ValueError("Expected a nonempty mono PCM signal at positive sample rate")
    x = samples.astype(np.float64)
    if samples.dtype.kind == "i":
        x /= float(2**(8*samples.dtype.itemsize-1))
    elif samples.dtype.kind == "u":
        center = float(2**(8*samples.dtype.itemsize-1))
        x = (x-center)/center
    x[1:] -= .97*x[:-1].copy()
    length, hop = int(rate*.025 + .5), int(rate*.010 + .5)
    frames = 1 + (max(0,len(x)-length)+hop-1)//hop
    padded = np.pad(x,(0,length+(frames-1)*hop-len(x)))
    framed = np.lib.stride_tricks.sliding_window_view(padded,length)[::hop] * np.hamming(length)
    nfft = 1 << (length-1).bit_length()
    spectrum = abs(np.fft.rfft(framed,n=nfft,axis=1))**2 / nfft
    energy = np.maximum(1e-12,spectrum @ filters(rate,nfft).T)
    coeff = dct(np.log(energy),type=2,norm="ortho",axis=1)[:,:13]
    positions = np.linspace(0,len(coeff)-1,16)
    lo = positions.astype(int)
    hi = np.minimum(lo+1,len(coeff)-1)
    t = (positions-lo)[:,None]
    return ((1-t)*coeff[lo]+t*coeff[hi]).ravel().astype(np.float32)


def read_corpus(path):
    with ZipFile(path) as archive:
        records = []
        for member in archive.namelist():
            name = PurePosixPath(member)
            if name.parent.name != "recordings" or name.suffix.lower() != ".wav":
                continue
            match = re.fullmatch(r"(\d+)_([^/]+)_(\d+)\.wav",name.name)
            if match is None:
                raise ValueError(f"Unrecognized audio filename {name.name}")
            digit, speaker, repeat = match.groups()
            records.append((speaker,int(digit),int(repeat),member))
        records.sort()
        ids = [row[:3] for row in records]
        people = sorted(set(r[0] for r in records))
        assert len(people)==6
        assert ids == [(sp,d,rep) for sp in people for d in range(10) for rep in range(50)]
        X = np.empty((len(ids),208),dtype=np.float32)
        rates, hashes = [], {}
        for i,(speaker,digit,repeat,member) in enumerate(records):
            rate, pcm = wavfile.read(io.BytesIO(archive.read(member)))
            rates.append(rate)
            wavehash = hashlib.sha256(str(rate).encode()+pcm.tobytes()).hexdigest()
            hashes.setdefault(wavehash,set()).add(speaker)
            X[i] = mfcc(pcm,rate)
    speakers = np.asarray([r[0] for r in ids])
    labels = np.asarray([r[1] for r in ids])
    names = np.asarray([PurePosixPath(r[-1]).name for r in records])
    manifest = dict(utterances=len(ids),speakers=people,rates=sorted(set(rates)),columns=208,
                    across_speaker_identical_waveforms=sum(len(owners)>1 for owners in hashes.values()),
                    matrix_sha256=hashlib.sha256(X.tobytes()).hexdigest())
    assert np.isfinite(X).all()
    return X,labels,speakers,names,manifest


def panels(X, y, speakers):
    people = np.unique(speakers)
    holdouts = []
    for i,seed in enumerate(SEEDS):
        mask = np.isin(speakers,people[2*i:2*i+2])
        train, test = np.flatnonzero(~mask),np.flatnonzero(mask)
        assert len(train)==2000 and len(test)==1000
        assert not np.intersect1d(speakers[train],speakers[test]).size
        holdouts.append(test)
        yield Panel("fsdd",i+1,seed,X[train],y[train],X[test],y[test],speakers[test]),train,test
    assert np.array_equal(np.sort(np.concatenate(holdouts)),np.arange(len(X)))
    assert all(not np.intersect1d(a,b).size for a,b in combinations(holdouts,2))
