"""Trajectory-level MD embedding and WT-restoration metrics.

This subpackage extends SAWNERGY from per-residue embeddings of a single
trajectory to **whole-trajectory embeddings that are comparable across
systems**, plus the statistics needed to tell a real system difference from
MD run-to-run noise.

The existing `sawnergy.embedding` trains a skip-gram model per frame, so each
frame lands in its own arbitrary basis (hence `align_frames`, an Orthogonal
Procrustes patch). That is fine for visualising one trajectory and unusable for
a wild-type-versus-mutant distance. Here a single encoder is shared across all
frames of all systems, so distances are well-posed without any alignment step.

Typical use::

    from sawnergy.md import (MdCathReader, run_descriptor, PCAEmbedder,
                             run_gates, WTReference, ThermalAxis, thermal_shift)

    descs = [run_descriptor(f) for f in reader.iter_runs(stride=5)]
    X = np.stack([d.vector() for d in descs])
    report = run_gates(X, systems, replicates, temperatures, corpus=corpus)
    print(report)

Nothing downstream should be believed until `run_gates` passes -- in
particular gate G3, the sham comparison, which checks that two halves of the
*same* system's replicates do **not** come out significantly different.
"""

from .manifest import (Corpus, CorpusError, CorpusReport, Independence, Role,
                       Run, MIN_REPLICATES, MIN_INDEPENDENCE)
from .readers import Frames, MdCathReader
from .features import (RunDescriptor, contact_map, frame_tensor,
                       frame_vectors, pair_distances, patch_frame_vectors,
                       patch_neighbourhoods, patch_vectors, run_descriptor,
                       seq_separation_mask)
from .atomic import (AtomTable, BURIAL_R, burial, cb_coords,
                     dihedral_features, mutation_mask, residue_min_distance,
                     sidechain_centroids)
from .temporal import TICA, autocorrelation, fluctuation, lifetime
from .states import (MarkovStates, TRANSITION_PRIOR, kmeans,
                     multinomial_noise)
from .stats import (Discriminability, VarianceComponents, autocorrelation_time,
                    leave_one_replicate_out, replicates_required,
                    separation_ratio, two_way_components, variance_components)
from .metrics import (DELTA_TM_K, ThermalAxis, ThermalShift, WTReference,
                      expected_floor, rescue_score, thermal_rescue_K,
                      thermal_shift)
from .encoder import (PCAEmbedder, TrainConfig, TrajectorySetEncoder,
                      build_frame_autoencoder, encode_frames,
                      train_frame_autoencoder)
from .autoencoder import (AEConfig, FitConfig, StageReport,
                          build_nested_autoencoder, build_time_encoding,
                          build_triplets, embed_runs, fit_nested, make_windows,
                          nested_loss, triplet_loss)
from .validate import (Gate, GateReport, replicate_splits, run_gates,
                       gate_corpus, gate_disentangled, gate_effect_size,
                       gate_features, gate_replicate_noise, gate_separability,
                       gate_sham, gate_temperature)
from .corpora import (DEFAULT_REPLICATES, P53_SYSTEMS, T_FULL, T_LADDER,
                      T_COLD, T_PHYSIOLOGICAL,
                      estimate_cost, method_validation_corpus, p53_corpus)

__all__ = [
    # manifest
    "Corpus", "CorpusError", "CorpusReport", "Independence", "Role", "Run",
    "MIN_REPLICATES", "MIN_INDEPENDENCE",
    # readers
    "Frames", "MdCathReader",
    # features
    "RunDescriptor", "contact_map", "frame_tensor", "frame_vectors",
    "pair_distances", "patch_frame_vectors", "patch_neighbourhoods",
    "patch_vectors",
    "run_descriptor", "seq_separation_mask",
    # atomic (all-atom residue geometry)
    "AtomTable", "BURIAL_R", "burial", "cb_coords", "dihedral_features",
    "mutation_mask", "residue_min_distance", "sidechain_centroids",
    # temporal
    "TICA", "autocorrelation", "fluctuation", "lifetime",
    # states (shared conformational state populations)
    "MarkovStates", "TRANSITION_PRIOR", "kmeans", "multinomial_noise",
    # stats
    "Discriminability", "VarianceComponents", "autocorrelation_time",
    "leave_one_replicate_out", "replicates_required", "separation_ratio",
    "two_way_components", "variance_components",
    # metrics
    "DELTA_TM_K", "ThermalAxis", "ThermalShift", "WTReference",
    "expected_floor", "rescue_score", "thermal_rescue_K", "thermal_shift",
    # encoder
    "PCAEmbedder", "TrainConfig", "TrajectorySetEncoder",
    "build_frame_autoencoder", "encode_frames", "train_frame_autoencoder",
    # autoencoder
    "AEConfig", "FitConfig", "StageReport", "build_nested_autoencoder",
    "build_time_encoding", "build_triplets", "embed_runs", "fit_nested",
    "make_windows", "triplet_loss",
    "nested_loss",
    # validate
    "Gate", "GateReport", "replicate_splits", "run_gates", "gate_corpus",
    "gate_disentangled", "gate_effect_size", "gate_features",
    "gate_replicate_noise", "gate_separability", "gate_sham",
    "gate_temperature",
    # corpora
    "DEFAULT_REPLICATES", "P53_SYSTEMS", "T_FULL", "T_LADDER", "T_COLD",
    "T_PHYSIOLOGICAL",
    "estimate_cost", "method_validation_corpus", "p53_corpus",
]
