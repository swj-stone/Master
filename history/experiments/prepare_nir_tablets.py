"""Recreate and verify the publisher's two IDRC shootout assay CSVs without extracting an archive."""
import io
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat

from experiments.fetch_nir_vote_data import ROOT
from experiments.stable_minpi_stress import file_hash


def main():
    manifest = pd.read_csv(ROOT / "nirbench-manifest.csv")
    specs = manifest[manifest.dataset.str.startswith("14-")].set_index("split")
    with ZipFile(ROOT / "nirbench-tablets.zip") as archive:
        names = [n for n in archive.namelist() if n.split("/")[-1] == "nir_shootout_2002.mat"]
        assert len(names) == 1
        data = loadmat(io.BytesIO(archive.read(names[0])), squeeze_me=True, struct_as_record=False)
    channels = list(map(str, data["calibrate_1"].axisscale[1, 0].ravel()))
    assay = [str(x).strip() for x in data["calibrate_Y"].label[1, 0].ravel()].index("assay")
    for split, groups in [("train", ("calibrate", "test")), ("test", ("validate",))]:
        row = specs.loc[split]
        X = np.vstack([data[group + "_1"].data for group in groups])
        y = np.concatenate([data[group + "_Y"].data[:, assay] for group in groups])
        frame = pd.DataFrame(X, columns=channels).assign(assay=np.round(y, 1))
        assert X.shape == (row.rows, row.spectral_features)
        dest = ROOT / "nirbench" / row.dataset / f"data_{split}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, lineterminator="\r\n")
        if file_hash(dest) != row.sha256 or dest.stat().st_size != row.bytes:
            raise ValueError(f"Generated {split} CSV differs from publisher manifest")
        print(split, len(frame), row.sha256)


if __name__ == "__main__":
    main()
