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

from megafold.app import app
from megafold.model.attention import Attend, Attention, full_pairwise_repr_to_windowed
from megafold.cli import cli
from megafold.configs import (
    ConductorConfig,
    MegaFoldConfig,
    TrainerConfig,
    create_megafold_from_yaml,
    create_trainer_from_conductor_yaml,
    create_trainer_from_yaml,
)
from megafold.inputs import (
    AtomDataset,
    AtomInput,
    BatchedAtomInput,
    MoleculeInput,
    MegaFoldInput,
    PDBDataset,
    PDBDistillationDataset,
    PDBInput,
    atom_input_to_file,
    collate_inputs_to_batched_atom_input,
    file_to_atom_input,
    maybe_transform_to_atom_input,
    maybe_transform_to_atom_inputs,
    megafold_input_to_biomolecule,
    megafold_inputs_to_batched_atom_input,
    pdb_dataset_to_atom_inputs,
    pdb_inputs_to_batched_atom_input,
    register_input_transform,
)
import os 

from megafold.model.megafold import (
    AdaptiveLayerNorm,
    AttentionPairBias,
    CentreRandomAugmentation,
    ComputeModelSelectionScore,
    ComputeRankingScore,
    ConditionWrapper,
    ConfidenceHead,
    ConfidenceHeadLogits,
    DiffusionModule,
    DiffusionTransformer,
    DistogramHead,
    ElucidatedAtomDiffusion,
    InputFeatureEmbedder,
    MSAModule,
    MSAPairWeightedAveraging,
    MultiChainPermutationAlignment,
    MegaFold,
    MegaFoldWithHubMixin,
    OuterProductMean,
    PairformerStack,
    PreLayerNorm,
    RelativePositionEncoding,
    TemplateEmbedder,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from megafold.trainer import DataLoader, Trainer
from megafold.utils.model_utils import (
    ComputeAlignmentError,
    ExpressCoordinatesInFrame,
    RigidFrom3Points,
    RigidFromReference3Points,
    SmoothLDDTLoss,
    weighted_rigid_align,
)

__all__ = [
    Attention,
    Attend,
    RelativePositionEncoding,
    RigidFrom3Points,
    RigidFromReference3Points,
    SmoothLDDTLoss,
    MultiChainPermutationAlignment,
    ExpressCoordinatesInFrame,
    ComputeAlignmentError,
    CentreRandomAugmentation,
    TemplateEmbedder,
    PreLayerNorm,
    AdaptiveLayerNorm,
    ConditionWrapper,
    OuterProductMean,
    MSAPairWeightedAveraging,
    TriangleMultiplication,
    AttentionPairBias,
    TriangleAttention,
    Transition,
    MSAModule,
    PairformerStack,
    DiffusionTransformer,
    DiffusionModule,
    ElucidatedAtomDiffusion,
    InputFeatureEmbedder,
    ConfidenceHead,
    DistogramHead,
    MegaFold,
    MegaFoldWithHubMixin,
    MegaFoldConfig,
    AtomInput,
    PDBInput,
    Trainer,
    TrainerConfig,
    ConductorConfig,
    ComputeRankingScore,
    ConfidenceHeadLogits,
    ComputeModelSelectionScore,
    create_megafold_from_yaml,
    create_trainer_from_yaml,
    create_trainer_from_conductor_yaml,
    pdb_inputs_to_batched_atom_input,
    weighted_rigid_align,
]
