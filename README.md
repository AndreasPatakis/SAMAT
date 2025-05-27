# Semantic-Aware Interpretable Multimodal Music Auto-Tagging

## Overview

This GitHub repository contains the code necessary to reproduce the results presented in the paper titled [Semantic-Aware Interpretable Multimodal Music Auto-Tagging](https://arxiv.org/abs/2505.17233).

## Data

The `data` folder contains all the datasets used in this work. Specifically, for the *MTG-Jamendo*, *Music4All*, and *AudioSet* datasets, you will find information about the train-validation-test splits, their features and values, as well as the category associated with each feature used.

## Executables

The following Python scripts execute the classification tasks described in the paper on all datasets, using the **EM-Banded** and **XGBoost** models. In addition to evaluation metrics, per-group importance results are computed, plotted, and saved.

- **jamendo_runs.py**: Runs all tasks on the *MTG-Jamendo* dataset.
- **m4a_runs.py**: Runs all tasks on the *Music4All* dataset.
- **audioset_runs.py**: Runs all tasks on the *AudioSet* dataset.

### Additional Files

- **embanded**: Contains the implementation of the **EM-Banded** algorithm, which is available in the official repository: [embanded](https://github.com/safugl/embanded).

- **helper.py**: Includes utility functions required for the execution of all experiments.

- **Supp.pdf**: Supplementary material with detailed feature descriptions and groupings.

## Citation

If you find the results of this paper useful, please consider citing it:

```bibtex
@misc{patakis2025semanticawareinterpretablemultimodalmusic,
  title        = {Semantic-Aware Interpretable Multimodal Music Auto-Tagging},
  author       = {Andreas Patakis and Vassilis Lyberatos and Spyridon Kantarelis and Edmund Dervakos and Giorgos Stamou},
  year         = {2025},
  eprint       = {2505.17233},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2505.17233}
}
