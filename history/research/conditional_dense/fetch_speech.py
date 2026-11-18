"""Fetch the original UCI archive and verify its SHA256 before storing it."""
import hashlib
from urllib.request import urlopen
from research.conditional_dense.speech import ARCHIVE, SHA256, SOURCE

if __name__=='__main__':
    if ARCHIVE.exists():
        if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()!=SHA256:
            raise ValueError("Existing archive hash differs")
    else:
        with urlopen(SOURCE,timeout=90) as stream: data=stream.read()
        if hashlib.sha256(data).hexdigest()!=SHA256: raise ValueError("Download hash differs")
        with ARCHIVE.open('xb') as out:out.write(data)
    print(f"Verified {ARCHIVE.name}: {SHA256}")
