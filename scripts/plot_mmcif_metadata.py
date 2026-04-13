# Copyright 2026 UIUC SSAIL
# Part of MegaFold project
#
# Licensed under the MIT License. See LICENSE file in the project root
# for full license text.
#
# If you use this code, please cite:
#
#   La, H., Gupta, A., Morehead, A., Cheng, J., & Zhang, M. (2025).
#   MegaFold: System-Level Optimizations for Accelerating Protein
#   Structure Prediction Models. arXiv:2506.20686
#   https://arxiv.org/abs/2506.20686
#
# BibTeX:
#   @misc{la2025megafoldsystemleveloptimizationsaccelerating,
#       title={MegaFold: System-Level Optimizations for Accelerating
#              Protein Structure Prediction Models},
#       author={Hoa La and Ahan Gupta and Alex Morehead
#               and Jianlin Cheng and Minjia Zhang},
#       year={2025},
#       eprint={2506.20686},
#       archivePrefix={arXiv},
#       primaryClass={q-bio.BM},
#       url={https://arxiv.org/abs/2506.20686},
#   }

import os

import plotly.express as px
import polars as pl


def plot_mmcif_metadata(mmcif_metadata_filepath: str, column: str):
    """Plots a distribution of the input mmCIF metadata."""
    df = pl.read_csv(mmcif_metadata_filepath)
    fig = px.histogram(df.to_pandas(), x=column, nbins=250)
    fig.show()


if __name__ == "__main__":
    plot_mmcif_metadata(
        os.path.join("caches", "pdb_data", "metadata", "mmcif.csv"),
        # os.path.join("caches", "afdb_data", "metadata", "mmcif.csv"),
        column="num_tokens",  # NOTE: must be one of the columns in the metadata file
    )
