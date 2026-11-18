# Strategic Feature Selection

This is the official code repository for [Strategic Feature Selection](https://arxiv.org/abs/2606.18867). The repository includes code for reproducing our experiments and implementing our proposed algorithm for jointly selecting the feature set and regularization level to combat strategic behavior.

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2606.18867" alt="arXiv"> <img src="https://img.shields.io/badge/paper-arXiv-red" /> </a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /> </a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"> <img src="https://img.shields.io/badge/license-MIT-blue.svg" /> </a>

</p>

## Setup

We provide an `environment.yaml` file with the Python dependencies. To create the conda environment, run the following commands from the root directory.

```bash
conda env create -f environment.yaml
conda activate strategic-selection
```

## Data generation

The data generation code is provided in
`upcoding_data_generation.ipynb` and can be run directly in Google
Colab. The baseline diagnosis data are 
simulated using the [upcoding R package](https://github.com/StanfordHPDS/upcoding). Place the generated data files under `data/upcoding/`:

- Simulated diagnosis data ZIP archive: `data/upcoding/upcoding_v28_top10_ha2025_export.zip`
- CMS-HCC V28 coefficients CSV file: `cms_v28_cna_coefficients_export_features.csv`

We also provide these files in the repository so the experiments can be
reproduced without rerunning data generation. The notebooks extract archives
when needed. 

We provide the synthetic benchmarks that we use to validate the algorithm in `data/lowdim_algorithm_benchmark_data.zip`.

## Reproducing the experiments

After setting up the environment and data, run the corresponding notebook for the desired figure from `notebooks/`. The generated figures are saved in the output directory specified by each notebook.

## Citation

```bibtex
@article{kaur2026strategic,
  title={Strategic Feature Selection},
  author={Kaur, Jivat Neet and Patil, Pratik and Shanmugam, Divya and Pierson, Emma  and Jordan, Michael I. and  Haghtalab, Nika and Jagadeesan, Meena and Alaa, Ahmed and Wang, Serena},
  journal={arXiv preprint arXiv:2606.18867},
  year={2026}
}
```
