# Dataset attribution

The UCI Machine Learning Repository lists the datasets below under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Each title links to the provider's citation and licensing record. The eight confirmation, four chain-validation and six witness-transfer archives are distributed as downloaded. Their split indices, selected columns and forecasts were produced for this study.

| Panel | Dataset | Creator(s) in the provider citation |
|---|---|---|
| Confirmation | [Glass Identification](https://doi.org/10.24432/C5WW2P) | B. German (1987) |
| Confirmation | [Raisin](https://doi.org/10.24432/C5660T) | İ. Çınar, M. Köklü and Ş. Taşdemir (2020) |
| Confirmation | [Wireless Indoor Localization](https://doi.org/10.24432/C51880) | R. Bhatt (2017) |
| Confirmation | [One-hundred Plant Species Leaves](https://doi.org/10.24432/C5RG76) | J. Cope, T. Beghin, P. Remagnino and S. Barman (2013) |
| Confirmation | [Airfoil Self-Noise](https://doi.org/10.24432/C5VW2C) | T. Brooks, D. Pope and M. Marcolini (1989) |
| Confirmation | [Yacht Hydrodynamics](https://doi.org/10.24432/C5XG7R) | J. Gerritsma, R. Onnink and A. Versluis (1981) |
| Confirmation | [Concrete Slump Test](https://doi.org/10.24432/C5FG7D) | I. Yeh (2007) |
| Confirmation | [Energy Efficiency](https://doi.org/10.24432/C51307) | A. Tsanas and A. Xifara (2012) |
| Chain validation | [Statlog (Landsat Satellite)](https://doi.org/10.24432/C55887) | A. Srinivasan (1993) |
| Chain validation | [QSAR Biodegradation](https://doi.org/10.24432/C5H60M) | K. Mansouri, T. Ringsted, D. Ballabio, R. Todeschini and V. Consonni (2013) |
| Chain validation | [QSAR Aquatic Toxicity](https://doi.org/10.24432/C5SG7H) | D. Ballabio, M. Cassotti, V. Consonni and R. Todeschini (2014) |
| Chain validation | [Combined Cycle Power Plant](https://doi.org/10.24432/C5002N) | P. Tüfekci and H. Kaya (2014) |
| Witness transfer | [MAGIC Gamma Telescope](https://doi.org/10.24432/C52C8B) | R. Bock (2004) |
| Witness transfer | [Musk (Version 1)](https://doi.org/10.24432/C5ZK5B) | D. Chapman and A. Jain (1994) |
| Witness transfer | [Wall-Following Robot Navigation Data](https://doi.org/10.24432/C57C8W) | A. Freire, M. Veloso and G. Barreto (2009) |
| Witness transfer | [Real Estate Valuation](https://doi.org/10.24432/C5J30W) | I. Yeh (2018) |
| Witness transfer | [Computer Hardware](https://doi.org/10.24432/C5830D) | J. Feldmesser (1987) |
| Witness transfer | [Forest Fires](https://doi.org/10.24432/C5D88D) | P. Cortez and A. Morais (2007) |
| Wider folds | [CNAE-9](https://doi.org/10.24432/C51G7P) | P. Ciarelli and E. Oliveira (2009) |
| Wider folds | [Ozone Level Detection](https://doi.org/10.24432/C5NG6W) | K. Zhang, W. Fan and X. Yuan (2008) |
| Wider folds | [Parkinson's Disease Classification](https://doi.org/10.24432/C5MS4X) | C. Sakar, G. Serbes, A. Gunduz, H. Nizam and B. Sakar (2018) |
| Wider folds | [SECOM](https://doi.org/10.24432/C54305) | M. McCann and A. Johnston (2008) |
| Wider folds | [Semeion Handwritten Digit](https://doi.org/10.24432/C5SC8V) | Semeion Handwritten Digit (1998); UCI lists no individual creator in its citation |
| Wider folds | [Urban Land Cover](https://doi.org/10.24432/C53S48) | B. Johnson (2013) |

For the wider panel, the release contains derived arrays and separate outer targets. The original preparation encodes classes, defines source-specific training and test partitions, and averages the three acoustic measurements per participant on Parkinson. The current comparison uses up to the first 100 columns. The [pinned source manifest](../../thesis-paper/evidence/wide/sources.json) gives checksums and credits. The [fold specifications](../../configs/context_review/wide_panel.json) give transformations and the use of dates, subjects and official splits for each source.

The pilot used the PMLB-formatted versions of [Ionosphere](https://doi.org/10.24432/C5W01B), by V. Sigillito, S. Wing, L. Hutton and K. Baker, [Statlog Vehicle Silhouettes](https://doi.org/10.24432/C5HG6N), by P. Mowforth and B. Shepherd, and [Sonar, Mines vs. Rocks](https://doi.org/10.24432/C5T01Q), by T. Sejnowski and R. Gorman. UCI lists CC BY 4.0 for each original dataset. The compressed PMLB tables and metadata are included as obtained from the fixed project revision. [PMLB release paper](https://doi.org/10.1093/bioinformatics/btab727) · [Pinned PMLB repository](https://github.com/EpistasisLab/pmlb/tree/7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68) · [PMLB MIT license notice](PMLB-LICENSE.txt)

The Diabetes pilot uses scikit-learn's packaged copy of the dataset described by B. Efron, T. Hastie, I. Johnstone and R. Tibshirani in *Least Angle Regression*. Its data file is supplied by the installed dependency. [Dataset description](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) · [Original paper](https://doi.org/10.1214/009053604000000067)
