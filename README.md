# Unimodal Strategies in Density-Based Clustering
<p align="center">
  <a href="https://ecmlpkdd.org/preprints/2025/" target="_blank">
    <img src="https://img.shields.io/badge/ECML--PKDD-2025-blue.svg" alt="ECML-PKDD 2025">
  </a>
  <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  </a>
  <img src="https://img.shields.io/badge/python-3.9%2B-yellow.svg" alt="Python 3.9+">
</p>

A **light-weight, parameter-free drop-in for DBSCAN** that finds its radius (ε) automatically in *O(D N²)* time by exploiting a near-unimodal relation between ε and the resulting number of clusters *k*.  
This repository accompanies the following paper, accepted to **ECML-PKDD 2025**:

> **Unimodal Strategies in Density-Based Clustering**  
> Oron Nir, Jay Tenenbaum, Ariel Shamir  
> ECML-PKDD 2025 (research track)

[[arXiv preprint]](https://arxiv.org/abs/2506.21695) | [[PDF]](Unimodal_Strategies-NirTenenbaumShamir-ECML-PKDD2025.pdf)

---

## Highlights

* **Near-Unimodality Discovery** – We show that *k(ε)* is quasi-unimodal for fixed *MinPts*.  
* **Ternary-Search (DBSCAN-TS)** – Quickly homes in on the mode ε\* with a handful of DBSCAN calls.  
* **TSE Estimator** – A faster sub-sampling variant with minimal accuracy loss.  
* Works out-of-the-box on **high-dimensional, large-scale** Vision, NLP, and Audio embeddings without dimensionality reduction.

---

## Quick Start

```bash
git clone https://github.com/oronnir/UnimodalStrategies.git
cd UnimodalStrategies
pip install -r requirements.txt     # numpy, scipy, scikit-learn, matplotlib
```

Unzip the evaluation dataset provided in `/ESC-50/ESC-50-training-CLAP_labeled_vectors.zip`. This zip contains two pickle files:
```
└─ ESC-50/
     └─ ESC-50-training-CLAP_labels.pkl    # class-label ids for ESC-50 training dataset
     └─ ESC-50-training-CLAP_vectors.pkl   # CLAP embeddings for ESC-50 training dataset     
```

## Logging Levels

- **ERROR**: Only shows error messages and exceptions
- **WARNING**: Shows warnings and errors
- **INFO**: Shows main progress information, results, and statistics (default)
- **DEBUG**: Shows detailed debugging information including all iterations and intermediate results

## Usage

### Setting Log Level via Environment Variable

**Windows (PowerShell):**
```powershell
$env:LOG_LEVEL="DEBUG"; python src/main.py
```

**Windows (Command Prompt):**
```cmd
set LOG_LEVEL=DEBUG && python src/main.py
```

**Linux/Mac:**
```bash
LOG_LEVEL=DEBUG python src/main.py
```

# Citation

For citation please use:
```
@inproceedings{nir2025unimodal,
  title     = {Unimodal Strategies in Density-Based Clustering},
  author    = {Nir, Oron and Tenenbaum, Jay and Shamir, Ariel},
  booktitle = {Proceedings of the European Conference on Machine Learning and Principles & Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year      = {2025},
  url       = {https://github.com/oronnir/UnimodalStrategies}
}
```
