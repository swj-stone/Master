"""Verify or retrieve the exact public input archives used by the text study."""
from experiments.document_vote_benchmark import ASSETS
from experiments.fetch_nir_vote_data import ROOT, download
from experiments.mail_vote_transfer import ARCHIVES


if __name__ == "__main__":
    uci = "https://archive.ics.uci.edu/static/public/"
    urls = {"sms": "228/sms+spam+collection.zip", "youtube": "380/youtube+spam+collection.zip",
            "sentences": "331/sentiment+labelled+sentences.zip"}
    for name, suffix in urls.items():
        download(uci + suffix, ROOT / f"text-{name}.zip", ASSETS[name])
    for folder, checksum in ARCHIVES.items():
        filename = f"20030228_{folder}.tar.bz2"
        download("https://spamassassin.apache.org/old/publiccorpus/" + filename, ROOT / filename, checksum)
