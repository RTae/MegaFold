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

import gzip
import os


def filter_pdb_lines(file_path: str, output_file_path: str):
    """Filter lines containing 'PDB' entries from a compressed `.dat.gz` file, and write them to an
    output file.

    :param file_path: Path to the compressed `.dat.gz` file to be read.
    :param output_file_path: Path to the output file where filtered lines will be written.
    """
    with gzip.open(file_path, "rt") as infile, open(output_file_path, "w") as outfile:
        # Run a generator expression to filter lines containing 'PDB'
        pdb_lines = (line for line in infile if "\tPDB\t" in line)
        outfile.writelines(pdb_lines)


if __name__ == "__main__":
    input_archive_file = "idmapping.dat.gz"
    output_file = os.path.join(
        "..", "data", "afdb_data", "data_caches", "uniprot_to_pdb_id_mapping.dat"
    )
    filter_pdb_lines(input_archive_file, output_file)
