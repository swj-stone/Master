"""Retrieve the frozen original Turkish SMS input for the revised vote study."""
from experiments.fetch_nir_vote_data import download
from experiments.refined_document_external import ASSET, SHA256


if __name__ == "__main__":
    download(
        "https://raw.githubusercontent.com/onrkrsy/TurkishSMS-Collection/main/TurkishSMSCollection.csv",
        ASSET, SHA256,
    )
