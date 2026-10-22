import functools
import inspect
import math
import threading
import warnings
from dataclasses import dataclass
import torch
import numpy as np
import pickle
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from warpgbm.metrics import (
    Metric,
    weighted_multi_mse_torch,
    weighted_rmsle_torch,
    weighted_log_loss_torch,
    weighted_accuracy_torch,
    weighted_corr_loss_torch,
)
from warpgbm.objectives import (
    Objective,
    Softmax,
    PER_OUTPUT,
    VECTOR,
    compute_grad_hess,
    resolve_objective,
    weighted_loss,
)
from warpgbm.binning import (
    MAX_NUM_BINS,
    MIN_NUM_BINS,
    _bin_matrix,
    _bin_matrix_cpu,
    _check_device,
    _prebinned_to_cpu,
    _prebinned_to_gpu,
    _quantile_edges,
    _to_float_gpu,
)
from warpgbm.cuda import _get_node_kernel


@dataclass
class _TargetWeights:
    """Effective weights and masks of a target set (internal)."""

    weight: torch.Tensor       # [N, K] normalized sample weight, 0 where the target is missing
    mask: torch.Tensor         # [N, K] bool, score entry has an observed target
    target_mask: torch.Tensor  # [N, T] bool, target entry is observed

    @property
    def row_weight(self) -> torch.Tensor:
        """[N] mean effective weight per row."""
        return self.weight.mean(dim=1)


# Width of a flattened tree node of flatten_tree (legacy layout of the v2.6 predict kernel)
_NODE_WIDTH = 7

# One CUDA graph capture at a time in this process: concurrent captures in several threads (fits
# running in parallel) break PyTorch's caching allocator (see WarpGBM._run_tree)
_GRAPH_CAPTURE_LOCK = threading.Lock()
_CAPTURE_MODE = "thread_local"
# Capture stream per device index. torch.cuda.graph's default capture stream is a single stream
# for the whole process, created on the device of the first capture: captures of models on other
# devices would run on it.
_CAPTURE_STREAMS = {}


def _capture_stream():
    """Side stream for CUDA graph captures on the current device (call with _GRAPH_CAPTURE_LOCK)."""
    index = torch.cuda.current_device()
    stream = _CAPTURE_STREAMS.get(index)
    if stream is None:
        stream = _CAPTURE_STREAMS[index] = torch.cuda.Stream(index)
    return stream


def _on_model_device(method):
    """
    Run a method with the model device as the current CUDA device (the caller's is restored).
    Kernels and streams that follow the current device (CUDA graph captures, events, PyTorch
    internals) then match the tensors when device="cuda:N" is not the thread's current device.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with torch.cuda.device(self.device):
            return method(self, *args, **kwargs)

    return wrapper

# Fraction of the free GPU memory (at fit) that the histograms of one tree level may use;
# above it a level is processed in node chunks without histogram subtraction
_HIST_MEMORY_FRACTION = 0.4

_REGRESSION_METRICS = ["mse", "corr", "rmsle", "loss"]
_MULTI_OUTPUT_METRICS = ["mse", "loss"]
_CLASSIFICATION_METRICS = ["logloss", "accuracy", "loss"]
_HIGHER_IS_BETTER = ("accuracy", "corr")
_RECORD_TRAINING_LOSS = ("eval", "all", "none")


class WarpGBM(BaseEstimator, RegressorMixin):
    # Defaults for instances pickled by versions without these attributes
    verbose = True
    _rng = None
    prebinned = None
    prebinned_ = False
    record_training_loss = "all"
    deterministic = True
    # Prediction caches (see _invalidate_prediction_cache)
    _pack_rounds = None
    _pack_concat = None
    _pack_device = None
    _pack_key = None
    _edges_cpu = None
    _base_cpu = None
    _base_gpu = None
    _flat_cpu = None
    _tree_graphs = {}
    # Replay trees as CUDA graphs (see _run_tree); False runs every tree eagerly
    _use_cuda_graphs = True

    def __init__(
        self,
        objective="regression",
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20,
        min_split_gain=0.0,
        L2_reg=1e-6,
        device="cuda",
        colsample_bytree=1.0,
        random_state=None,
        warm_start=False,
        monotonic_constraints=None,
        min_sum_hessian=1e-3,
        verbose=True,
        prebinned=None,
        record_training_loss="eval",
        deterministic=True,
    ):
        # Validate arguments
        self._validate_hyperparams(
            objective=objective,
            num_bins=num_bins,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            L2_reg=L2_reg,
            device=device,
            colsample_bytree=colsample_bytree,
            monotonic_constraints=monotonic_constraints,
            min_sum_hessian=min_sum_hessian,
            prebinned=prebinned,
            record_training_loss=record_training_loss,
            deterministic=deterministic,
        )

        self.objective = objective
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.forest = None
        self.bin_edges = None
        self.base_prediction = None
        self.unique_eras = None
        self.device = device
        self.num_classes = None
        self.num_outputs = None
        self.classes_ = None
        self.objective_ = None
        self.scores = None
        self.root_node_indices = None
        self.bin_indices = None
        self.Y_gpu = None
        self.num_features = None
        self.num_samples = None
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.min_sum_hessian = min_sum_hessian
        self.L2_reg = L2_reg
        self.forest = [{} for _ in range(self.n_estimators)]
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.warm_start = warm_start
        self.monotonic_constraints = monotonic_constraints
        self.verbose = verbose
        self.prebinned = prebinned
        self.record_training_loss = record_training_loss
        self.deterministic = deterministic
        self.prebinned_ = False
        self._rng = None
        self.label_encoder = None
        self.feature_importance_ = None
        self.per_era_feature_importance_ = None
        self.monotonic_constraints_vector_ = None
        self.monotonic_required_split_direction_ = None
        self.monotonic_required_split_direction_gpu_ = None
        self._monotonic_constraints_fitted_ = False
        self._targets_1d = True
        self._is_fitted = False
        self._trees_trained = 0  # Track number of trees already trained
        self._invalidate_prediction_cache()
        self._objective_source = None
        self._reset_eval_state()
        self._per_era_importance_all = None
        self._per_era_importance_best = None
        self.train_predictions_ = None

    def _validate_hyperparams(self, **kwargs):
        # All kernels are CUDA-only; there is no CPU code path
        _check_device(kwargs["device"])

        # Validate objective
        objective = kwargs["objective"]
        if isinstance(objective, str):
            if objective not in [
                "regression",
                "multi_regression",
                "multiclass",
                "binary",
            ]:
                raise ValueError(
                    "objective must be 'regression', 'multi_regression', 'binary', "
                    f"'multiclass', an Objective instance or a callable, got {objective}."
                )
        elif not isinstance(objective, Objective) and not callable(objective):
            raise ValueError(
                "objective must be 'regression', 'multi_regression', 'binary', "
                f"'multiclass', an Objective instance or a callable, got {objective!r}."
            )

        # Type checks
        int_params = [
            "num_bins",
            "max_depth",
            "n_estimators",
        ]
        float_params = [
            "learning_rate",
            "min_child_weight",
            "min_split_gain",
            "min_sum_hessian",
            "L2_reg",
            "colsample_bytree",
        ]

        for param in int_params:
            if not isinstance(kwargs[param], int):
                raise TypeError(
                    f"{param} must be an integer, got {type(kwargs[param])}."
                )

        for param in float_params:
            if not isinstance(
                kwargs[param], (float, int)
            ):  # Accept ints as valid floats
                raise TypeError(f"{param} must be a float, got {type(kwargs[param])}.")

        if not (MIN_NUM_BINS <= kwargs["num_bins"] <= MAX_NUM_BINS):
            raise ValueError(
                f"num_bins must be between {MIN_NUM_BINS} and {MAX_NUM_BINS} inclusive."
            )
        if kwargs["max_depth"] < 1:
            raise ValueError("max_depth must be at least 1.")
        if not (0.0 < kwargs["learning_rate"] <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0].")
        if kwargs["n_estimators"] <= 0:
            raise ValueError("n_estimators must be positive.")
        if kwargs["min_child_weight"] < 0:
            raise ValueError("min_child_weight must be non-negative.")
        if kwargs["min_split_gain"] < 0:
            raise ValueError("min_split_gain must be non-negative.")
        if kwargs["min_sum_hessian"] < 0:
            raise ValueError("min_sum_hessian must be non-negative.")
        if kwargs["L2_reg"] < 0:
            raise ValueError("L2_reg must be non-negative.")
        if kwargs["colsample_bytree"] <= 0 or kwargs["colsample_bytree"] > 1:
            raise ValueError(
                f"Invalid colsample_bytree: {kwargs['colsample_bytree']}. Must be a float value > 0 and <= 1."
            )

        if kwargs["prebinned"] not in (None, True, False):
            raise TypeError(f"prebinned must be None, True or False, got {kwargs['prebinned']!r}.")
        if not isinstance(kwargs["deterministic"], (bool, np.bool_)):
            raise TypeError(f"deterministic must be True or False, got {kwargs['deterministic']!r}.")
        if kwargs["record_training_loss"] not in _RECORD_TRAINING_LOSS:
            raise ValueError(
                f"record_training_loss must be one of {_RECORD_TRAINING_LOSS}, "
                f"got {kwargs['record_training_loss']!r}."
            )

        mc = kwargs.get("monotonic_constraints")
        if mc is not None and not isinstance(mc, (list, tuple, np.ndarray, dict)):
            raise TypeError(
                "monotonic_constraints must be None, a vector-like object, or a dict {feature_idx: -1|0|1}."
            )

    def _normalize_monotonic_constraints(self, n_features):
        constraints = self.monotonic_constraints

        if constraints is None:
            return np.zeros(n_features, dtype=np.int8)

        if isinstance(constraints, dict):
            dense = np.zeros(n_features, dtype=np.int8)
            for key, value in constraints.items():
                if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
                    raise TypeError(
                        "monotonic_constraints dict keys must be integer feature indices."
                    )
                if key < 0 or key >= n_features:
                    raise ValueError(
                        f"monotonic_constraints key {key} out of range for {n_features} features."
                    )
                if value not in (-1, 0, 1):
                    raise ValueError(
                        "monotonic_constraints values must be one of {-1, 0, 1}."
                    )
                dense[int(key)] = int(value)
            return dense

        dense = np.asarray(constraints, dtype=np.int8)
        if dense.ndim != 1:
            raise ValueError(
                f"monotonic_constraints vector must be 1-dimensional, got shape {dense.shape}."
            )
        if dense.shape[0] != n_features:
            raise ValueError(
                f"monotonic_constraints vector length must equal n_features ({n_features}), got {dense.shape[0]}."
            )
        if not np.isin(dense, [-1, 0, 1]).all():
            raise ValueError("monotonic_constraints values must be one of {-1, 0, 1}.")
        return dense.astype(np.int8, copy=False)

    def _prepare_monotonic_constraints(self, n_features):
        dense = self._normalize_monotonic_constraints(n_features)

        if self.warm_start and self._is_fitted and self._monotonic_constraints_fitted_:
            if not np.array_equal(dense, self.monotonic_constraints_vector_):
                raise ValueError(
                    "monotonic_constraints must match the fitted model when warm_start=True."
                )

        self.monotonic_constraints_vector_ = dense
        # Split direction convention: +1 means left leaf > right leaf.
        # For monotonic increasing features, right should be >= left, hence required split dir = -1.
        self.monotonic_required_split_direction_ = -dense
        self.monotonic_required_split_direction_gpu_ = (
            torch.from_numpy(self.monotonic_required_split_direction_)
            .to(torch.int8)
            .to(self.device)
        )
        self._monotonic_constraints_fitted_ = True

    # ─── Objective / shape helpers ───

    @property
    def num_trees_(self):
        """Number of trained boosting rounds (per_output models have K trees per round)."""
        return self._trees_trained

    @property
    def nan_bin(self):
        """
        Internal bin index of missing (NaN) feature values. It is a label, not an ordering:
        splits route it by their learned nan_left flag. Pre-binned input uses -1 instead.
        """
        return self.num_bins

    def _resolve_objective(self):
        if not (self.warm_start and self._is_fitted and self.objective_ is not None):
            self._objective_source = self.objective
            return resolve_objective(self.objective)

        source = self._objective_source
        unchanged = self.objective is source or (
            isinstance(self.objective, str) and self.objective == source
        )
        if not unchanged:
            raise ValueError(
                "The objective cannot be changed when continuing training with warm_start=True."
            )
        return self.objective_

    def _reset_eval_state(self):
        self.best_iteration_ = None
        self.best_score_ = None
        self.evals_result_ = {"train": [], "eval": []}
        self.eval_iterations_ = []
        self.eval_loss = self.evals_result_["eval"]  # backward-compatible alias
        self.eval_metric_name_ = None
        self.eval_metric_higher_is_better_ = None
        self._rollback = False

    def _default_num_trees(self):
        """Rounds used by default: the best round after early stopping, else all trained rounds."""
        if self._rollback and self.best_iteration_ is not None:
            return self.best_iteration_
        return self._trees_trained

    def _resolve_num_trees(self, num_trees):
        if num_trees is None:
            return self._default_num_trees()
        if isinstance(num_trees, bool) or not isinstance(num_trees, (int, np.integer)):
            raise TypeError(f"num_trees must be an int or None, got {num_trees!r}.")
        if not 0 <= num_trees <= self._trees_trained:
            raise ValueError(
                f"num_trees must be between 0 and the number of trained rounds "
                f"({self._trees_trained}), got {num_trees}."
            )
        return int(num_trees)

    def _is_classifier(self):
        obj = self.objective_
        if obj is None:
            obj = resolve_objective(self.objective)
        return obj.is_classifier

    def _output_strategy(self):
        return self.objective_.output_strategy

    def _base_score_tensor(self):
        """Initial raw score as a [K] tensor on the model device (cached, see _base_score_np)."""
        if self._base_gpu is None or self._base_gpu.device != torch.device(self.device):
            self._base_gpu = self._to_device(self._base_score_np())
        return self._base_gpu

    def _to_device(self, array):
        """numpy array -> tensor on the model device, copied asynchronously from pinned memory."""
        return torch.from_numpy(np.ascontiguousarray(array)).pin_memory().to(
            self.device, non_blocking=True
        )

    def _store_base_score(self, base):
        """Keep the legacy representation: a float for single-output regression."""
        self._base_cpu = None
        self._base_gpu = None
        base = base.detach().to(torch.float32).reshape(-1)
        if base.numel() == 1 and not self.objective_.is_classifier:
            self.base_prediction = float(base.item())
        else:
            self.base_prediction = base.to(self.device)

    def validate_fit_params(
        self,
        X,
        y,
        era_id,
        X_eval,
        y_eval,
        eval_every_n_trees,
        early_stopping_rounds,
        eval_metric,
        sample_weight,
        sample_weight_eval,
        eval_aux=None,
        train_aux=None,
    ):
        obj = self.objective_

        # ─── Required: X and y ───
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

        self._validate_target_ndim(y, "y")
        if y.ndim == 2 and y.shape[1] < 1:
            raise ValueError("y must contain at least one output column.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows. Got {X.shape[0]} and {y.shape[0]}."
            )

        # ─── Optional: sample_weight ───
        if sample_weight is not None:
            self._validate_weight(sample_weight, y, "sample_weight")

        if sample_weight_eval is not None and (X_eval is None or y_eval is None):
            raise ValueError(
                "sample_weight_eval can only be used when X_eval and y_eval are provided."
            )

        # ─── Optional: era_id ───
        if era_id is not None:
            if not isinstance(era_id, np.ndarray):
                raise TypeError("era_id must be a numpy array.")
            if era_id.ndim != 1:
                raise ValueError(
                    f"era_id must be 1-dimensional, got shape {era_id.shape}"
                )
            if len(era_id) != X.shape[0]:
                raise ValueError(
                    f"era_id must have same length as y. Got {len(era_id)} and {X.shape[0]}."
                )

        # ─── Optional: Eval Set ───
        eval_args = [X_eval, y_eval, eval_every_n_trees]
        if any(arg is not None for arg in eval_args):
            # Require all of them
            if X_eval is None or y_eval is None or eval_every_n_trees is None:
                raise ValueError(
                    "If using eval set, X_eval, y_eval, and eval_every_n_trees must all be defined."
                )

            if not isinstance(X_eval, np.ndarray) or not isinstance(y_eval, np.ndarray):
                raise TypeError("X_eval and y_eval must be numpy arrays.")
            if X_eval.ndim != 2:
                raise ValueError(
                    f"X_eval must be 2-dimensional, got shape {X_eval.shape}"
                )

            self._validate_target_ndim(y_eval, "y_eval")
            if y_eval.ndim != y.ndim or (
                y.ndim == 2 and y_eval.shape[1] != y.shape[1]
            ):
                raise ValueError(
                    f"y_eval must have the same number of outputs as y. Got shapes {y_eval.shape} and {y.shape}."
                )

            if X_eval.shape[0] != y_eval.shape[0]:
                raise ValueError(
                    f"X_eval and y_eval must have same number of rows. Got {X_eval.shape[0]} and {y_eval.shape[0]}."
                )

            if sample_weight_eval is not None:
                self._validate_weight(sample_weight_eval, y_eval, "sample_weight_eval")

            if not isinstance(eval_every_n_trees, int) or eval_every_n_trees <= 0:
                raise ValueError(
                    f"eval_every_n_trees must be a positive integer, got {eval_every_n_trees}."
                )

            if early_stopping_rounds is not None:
                if (
                    not isinstance(early_stopping_rounds, int)
                    or early_stopping_rounds <= 0
                ):
                    raise ValueError(
                        f"early_stopping_rounds must be a positive integer, got {early_stopping_rounds}."
                    )
            else:
                # No early stopping = set to "never trigger"
                early_stopping_rounds = self.n_estimators + 1

            if eval_aux is not None:
                self._validate_aux(eval_aux, X_eval.shape[0], "eval_aux", "X_eval", eval_metric)

            if eval_metric is not None and not callable(eval_metric):
                if obj.is_classifier:
                    valid_metrics = _CLASSIFICATION_METRICS + _REGRESSION_METRICS
                elif y.ndim == 2:
                    valid_metrics = _MULTI_OUTPUT_METRICS
                else:
                    valid_metrics = _REGRESSION_METRICS
                if eval_metric not in valid_metrics:
                    raise ValueError(
                        f"Invalid eval_metric: {eval_metric}. Choose from {valid_metrics} or pass a callable."
                    )
                if eval_metric == "loss" and not obj.has_loss():
                    raise ValueError(
                        f"eval_metric='loss' requires the objective {obj!r} to implement loss()."
                    )

        if eval_aux is not None and (X_eval is None or y_eval is None):
            raise ValueError("eval_aux can only be used when X_eval and y_eval are provided.")
        if train_aux is not None:
            if X_eval is None or y_eval is None:
                raise ValueError("train_aux can only be used when X_eval and y_eval are provided.")
            self._validate_aux(train_aux, X.shape[0], "train_aux", "X", eval_metric)

        return early_stopping_rounds  # May have been defaulted here

    @staticmethod
    def _aux_mode(metric):
        """How a callable metric takes ``aux``: "required", "optional" or None (not accepted)."""
        fn = metric.fn if isinstance(metric, Metric) else metric
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return None
        if "aux" in params:
            return "optional" if params["aux"].default is not inspect.Parameter.empty else "required"
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return "optional"
        return None

    def _validate_aux(self, aux, n_rows, name, x_name, eval_metric):
        if not isinstance(aux, np.ndarray) or aux.ndim not in (1, 2):
            raise TypeError(f"{name} must be a 1- or 2-dimensional numpy array.")
        if aux.shape[0] != n_rows:
            raise ValueError(
                f"{name} must have the same number of rows as {x_name}. Got {aux.shape[0]} and {n_rows}."
            )
        if not callable(eval_metric):
            raise ValueError(f"{name} requires a callable eval_metric that accepts an `aux` argument.")
        if self._aux_mode(eval_metric) is None:
            raise ValueError(
                f"{name} was given, but eval_metric does not accept an `aux` argument."
            )

    def _validate_target_ndim(self, y, name):
        if self.objective == "multi_regression":
            if y.ndim != 2:
                raise ValueError(
                    f"{name} must be 2-dimensional for objective='multi_regression', got shape {y.shape}"
                )
        elif isinstance(self.objective, str) or self.objective_.is_classifier:
            if y.ndim != 1:
                raise ValueError(f"{name} must be 1-dimensional, got shape {y.shape}")
        elif y.ndim not in (1, 2):
            raise ValueError(
                f"{name} must be 1- or 2-dimensional for a custom objective, got shape {y.shape}"
            )

    def _validate_weight(self, weight_np, y, name):
        if not isinstance(weight_np, np.ndarray):
            raise TypeError(f"{name} must be a numpy array.")

        is_multi_output = y.ndim == 2
        if is_multi_output:
            self._validate_multi_output_weight_shape(weight_np, y.shape, name)
        else:
            if weight_np.ndim != 1:
                raise ValueError(
                    f"{name} must be 1-dimensional, got shape {weight_np.shape}"
                )
            if len(weight_np) != len(y):
                raise ValueError(
                    f"{name} must have same length as y. Got {len(weight_np)} and {len(y)}."
                )

        if not np.all(np.isfinite(weight_np)):
            raise ValueError(f"{name} must contain only finite values.")
        if is_multi_output:
            if not np.all(weight_np >= 0):
                raise ValueError(f"{name} must be non-negative.")
            if not np.any(weight_np > 0):
                raise ValueError(f"{name} must contain at least one positive value.")
        else:
            if not np.all(weight_np > 0):
                raise ValueError(f"{name} must be strictly positive.")

    def _validate_multi_output_weight_shape(self, weight_np, y_shape, weight_name):
        n_samples, n_outputs = y_shape
        if weight_np.ndim == 1:
            if weight_np.shape[0] != n_samples:
                raise ValueError(
                    f"{weight_name} must have same length as y rows. Got {weight_np.shape[0]} and {n_samples}."
                )
            return

        if weight_np.ndim == 2:
            if weight_np.shape != (n_samples, n_outputs):
                raise ValueError(
                    f"{weight_name} must have shape {(n_samples, n_outputs)} for multi-output targets, got {weight_np.shape}."
                )
            return

        raise ValueError(
            f"{weight_name} must be 1-dimensional or 2-dimensional, got shape {weight_np.shape}."
        )

    def _normalize_sample_weight(self, weight_np: np.ndarray) -> np.ndarray:
        weight_np = weight_np.astype(np.float32, copy=False)
        mean_w = float(weight_np.mean())
        if mean_w <= 0:
            raise ValueError("sample weights must have positive mean.")
        return weight_np / mean_w

    def _build_targets(self, y_prepared, sample_weight):
        """
        Build GPU targets with their effective weights and masks.

        Args:
            y_prepared: [N, T] float32 numpy targets, NaN/inf = missing.
            sample_weight: None, [N] or [N, T] numpy weights (already normalized).

        Returns:
            (Y_gpu [N, T] with missing entries filled with 0, _TargetWeights)
        """
        n, t = y_prepared.shape
        K = self.num_outputs
        target_mask_np = np.isfinite(y_prepared)
        y_filled = np.where(target_mask_np, y_prepared, 0.0).astype(np.float32)

        Y_gpu = torch.from_numpy(y_filled).to(self.device)
        target_mask = torch.from_numpy(target_mask_np).to(self.device)
        mask = self.objective_.score_mask(target_mask, K).contiguous()

        if sample_weight is None:
            weight = torch.ones((n, K), dtype=torch.float32, device=self.device)
        else:
            w = torch.from_numpy(
                np.ascontiguousarray(sample_weight, dtype=np.float32)
            ).to(self.device)
            if w.ndim == 1:
                weight = w.view(-1, 1).expand(n, K)
            elif w.shape[1] == K:
                weight = w
            else:
                raise ValueError(
                    f"2-D sample weights with {w.shape[1]} columns cannot be mapped to {K} score columns."
                )
        weight = torch.where(mask, weight, torch.zeros((), device=self.device))

        tw = _TargetWeights(
            weight=weight.to(torch.float32).contiguous(),
            mask=mask,
            target_mask=target_mask,
        )
        return Y_gpu, tw

    # ─── Packed forest (prediction) ───
    #
    # Every boosting round is flattened once into numpy arrays (see warpgbm/cuda/predict.cu):
    #   nodes [t, M, 4] int32: feature (-1 = leaf), split_bin | nan_left << 8, left, right
    #   leaf  [t, M, Lw] float32: leaf values times the learning rate (Lw = K for vector trees)
    #   out_col [t] int32: score column of each tree (per-output forests only)
    # where t is the number of trees of the round. Rounds are packed lazily and kept until the
    # forest changes (fresh fit, truncate, load); the first n rounds are a prefix of the
    # concatenation, so any num_trees is a slice and never re-flattens.

    def _invalidate_prediction_cache(self, keep_rounds=0):
        """Drop packed rounds from keep_rounds on, and all concatenations."""
        packs = self._pack_rounds
        self._pack_rounds = [] if not packs else packs[:keep_rounds]
        self._pack_concat = None
        self._pack_device = None
        self._flat_cpu = None
        if keep_rounds == 0:
            self._pack_key = None
            self._edges_cpu = None
            self._base_cpu = None
            self._base_gpu = None

    def _pack_layout(self):
        """(max nodes per tree, leaf width, per-output) of the packed forest."""
        per_output = self._output_strategy() == PER_OUTPUT
        leaf_width = 1 if per_output else self.num_outputs
        return 2 ** (self.max_depth + 1), leaf_width, per_output

    def _pack_tree(self, tree, max_nodes, leaf_width):
        """Flatten one tree dict (pre-order node ids) into (nodes [M, 4], leaf [M, Lw])."""
        nodes = np.zeros((max_nodes, 4), dtype=np.int32)
        nodes[:, 0] = -1
        leaf = np.zeros((max_nodes, leaf_width), dtype=np.float32)
        counter = 0
        stack = [(tree, None, 0)]  # (node, parent id, 2 = left / 3 = right)
        while stack:
            node, parent, side = stack.pop()
            i = counter
            counter += 1
            if parent is not None:
                nodes[parent, side] = i
            if "leaf_value" in node:
                value = node["leaf_value"]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().tolist()
                if isinstance(value, (list, tuple, np.ndarray)):
                    leaf[i] = np.asarray(value, dtype=np.float32)[:leaf_width]
                else:
                    leaf[i] = value
            else:
                nodes[i, 0] = int(node["feature"])
                nodes[i, 1] = int(node["bin"]) | (int(bool(node.get("nan_left", False))) << 8)
                # Right is pushed first so that the left subtree gets the next ids
                stack.append((node["right"], i, 3))
                stack.append((node["left"], i, 2))
        leaf *= np.float32(self.learning_rate)
        return nodes, leaf

    def _pack_round(self, round_trees):
        max_nodes, leaf_width, per_output = self._pack_layout()
        trees = round_trees if per_output else [round_trees]
        packed = [
            (self._pack_tree(tree, max_nodes, leaf_width), k)
            for k, tree in enumerate(trees)
            if tree
        ]
        nodes = np.zeros((len(packed), max_nodes, 4), dtype=np.int32)
        leaf = np.zeros((len(packed), max_nodes, leaf_width), dtype=np.float32)
        out_col = np.zeros(len(packed), dtype=np.int32)
        for j, ((n, l), k) in enumerate(packed):
            nodes[j], leaf[j], out_col[j] = n, l, k
        return nodes, leaf, out_col

    def _ensure_packed(self, num_rounds):
        key = (float(self.learning_rate), self.max_depth, self.num_outputs, self._output_strategy())
        if self._pack_key != key:
            self._invalidate_prediction_cache()
            self._pack_key = key
        for r in range(len(self._pack_rounds), num_rounds):
            self._pack_rounds.append(self._pack_round(self.forest[r]))

    def _packed_forest(self, num_rounds, device=None):
        """
        Packed first num_rounds rounds: (nodes, leaf, out_col or None, round_end) as numpy
        arrays (device=None) or tensors on device. round_end[r] = trees in rounds 0..r.
        """
        self._ensure_packed(num_rounds)
        if self._pack_concat is None or self._pack_concat[0] < num_rounds:
            packs = self._pack_rounds
            max_nodes, leaf_width, per_output = self._pack_layout()
            if packs:
                nodes = np.concatenate([p[0] for p in packs])
                leaf = np.concatenate([p[1] for p in packs])
                out_col = np.concatenate([p[2] for p in packs])
            else:
                nodes = np.zeros((0, max_nodes, 4), dtype=np.int32)
                leaf = np.zeros((0, max_nodes, leaf_width), dtype=np.float32)
                out_col = np.zeros(0, dtype=np.int32)
            round_end = np.cumsum([len(p[0]) for p in packs], dtype=np.int64)
            self._pack_concat = (len(packs), nodes, leaf, out_col if per_output else None, round_end)
            self._pack_device = None
        _, nodes, leaf, out_col, round_end = self._pack_concat
        n_trees = int(round_end[num_rounds - 1]) if num_rounds > 0 else 0
        if device is None:
            return (
                nodes[:n_trees],
                leaf[:n_trees],
                None if out_col is None else out_col[:n_trees],
                round_end[:num_rounds],
            )
        if self._pack_device is None or self._pack_device[0] != (self._pack_concat[0], str(device)):
            self._pack_device = (
                (self._pack_concat[0], str(device)),
                self._to_device(nodes),
                self._to_device(leaf),
                None if out_col is None else self._to_device(out_col),
            )
        _, nodes_t, leaf_t, out_col_t = self._pack_device
        return (
            nodes_t[:n_trees],
            leaf_t[:n_trees],
            None if out_col_t is None else out_col_t[:n_trees],
            round_end[:num_rounds],
        )

    def _forest_kernel(self, bins, nodes, leaf, out_col, base, tree_counts, feature_major=False):
        """
        Raw scores [P, N, K] = base + sum of the first tree_counts[p] packed trees, on the GPU.
        bins is row-major [N, F] or feature-major [F, N] (feature_major=True).
        """
        if feature_major:
            num_rows = bins.size(1)
            row_stride, feat_stride = 1, bins.size(1)
        else:
            num_rows = bins.size(0)
            row_stride, feat_stride = bins.size(1), 1
        ckpt = self._to_device(np.asarray(tree_counts, dtype=np.int64))
        out = torch.empty(
            (len(tree_counts), num_rows, self.num_outputs), device=self.device, dtype=torch.float32
        )
        _get_node_kernel().predict_forest(
            bins.contiguous(),
            row_stride,
            feat_stride,
            num_rows,
            nodes,
            leaf,
            out_col,
            base.contiguous(),
            self.nan_bin,
            ckpt,
            out,
        )
        return out

    def _predict_raw_staged_binned(self, bin_indices, rounds, feature_major=False, base=None):
        """
        Raw scores [P, N, K] after rounds[p] boosting rounds (rounds ascending), on the GPU.
        base: [K] score the trees are added to (default: the model's base score).
        """
        nodes, leaf, out_col, round_end = self._packed_forest(max(rounds), self.device)
        tree_counts = [int(round_end[r - 1]) if r > 0 else 0 for r in rounds]
        if base is None:
            base = self._base_score_tensor()
        return self._forest_kernel(
            bin_indices, nodes, leaf, out_col, base, tree_counts, feature_major
        )

    def _flat_forest_cpu(self, num_rounds):
        """
        Packed forest as flat per-field arrays over global node ids t * M + j, with the children
        as global ids (cached with the packed forest): (feature, split_bin, nan_left, left, right,
        leaf [T * M, Lw], out_col, round_end).
        """
        _, _, out_col, round_end = self._packed_forest(num_rounds)
        # Flatten the whole concatenation once; a prefix of rounds only visits its own trees
        key, nodes, leaf = self._pack_concat[:3]
        flat = self._flat_cpu
        if flat is None or flat[0] != key:
            T, M = nodes.shape[:2]
            offset = (np.arange(T, dtype=np.int64) * M)[:, None]
            flat = (
                key,
                nodes[..., 0].reshape(-1),
                (nodes[..., 1] & 0xFF).reshape(-1),
                ((nodes[..., 1] >> 8) & 1).astype(bool).reshape(-1),
                (nodes[..., 2] + offset).reshape(-1),
                (nodes[..., 3] + offset).reshape(-1),
                leaf.reshape(T * M, leaf.shape[2]),
            )
            self._flat_cpu = flat
        return flat[1:] + (out_col, round_end)

    def _predict_raw_staged_cpu(self, bins, rounds):
        """CPU version of _predict_raw_staged_binned for internal bins [N, F] (numpy int8)."""
        feature, split_bin, nan_left, left, right, leaf, out_col, round_end = self._flat_forest_cpu(
            max(rounds)
        )
        tree_counts = [int(round_end[r - 1]) if r > 0 else 0 for r in rounds]
        base = self._base_score_np()
        K = self.num_outputs
        n, T = bins.shape[0], max(tree_counts)
        out = np.empty((len(rounds), n, K), dtype=np.float32)
        if T == 0 or n == 0:
            out[:] = base
            return out
        M = 2 ** (self.max_depth + 1)
        roots = np.arange(T, dtype=np.int64) * M
        F = bins.shape[1]
        # Bound the [rows, T, K] intermediate arrays
        chunk = max(1, (1 << 22) // (T * max(K, 4)))
        for start in range(0, n, chunk):
            b = bins[start : start + chunk]
            rows = b.shape[0]
            flat_bins = b.reshape(-1).astype(np.int64)
            row_offset = (np.arange(rows, dtype=np.int64) * F)[:, None]
            node = np.broadcast_to(roots, (rows, T)).copy()
            for _ in range(self.max_depth + 1):
                f = feature[node]
                is_leaf = f < 0
                if is_leaf.all():
                    break
                x = flat_bins[row_offset + np.maximum(f, 0)]
                go_left = np.where(x == self.nan_bin, nan_left[node], x <= split_bin[node])
                node = np.where(is_leaf, node, np.where(go_left, left[node], right[node]))
            values = leaf[node]  # [rows, T, Lw]
            if out_col is not None:
                dense = np.zeros((rows, T, K), dtype=np.float32)
                dense[:, np.arange(T), out_col[:T]] = values[..., 0]
                values = dense
            # Sequential float32 sums over the trees, the same additions as the CUDA kernel
            cum = np.add.accumulate(values, axis=1)
            for p, t in enumerate(tree_counts):
                acc = cum[:, t - 1] if t > 0 else np.zeros((rows, K), dtype=np.float32)
                out[p, start : start + chunk] = base + acc
        return out

    # ─── Fit ───

    @_on_model_device
    def fit(
        self,
        X,
        y,
        era_id=None,
        sample_weight=None,
        X_eval=None,
        y_eval=None,
        sample_weight_eval=None,
        eval_every_n_trees=None,
        early_stopping_rounds=None,
        eval_metric=None,
        eval_aux=None,
        train_aux=None,
    ):
        self.objective_ = self._resolve_objective()
        has_eval_set = X_eval is not None and y_eval is not None
        # Roll back to the best evaluated round only when early stopping was requested
        rollback = has_eval_set and early_stopping_rounds is not None

        early_stopping_rounds = self.validate_fit_params(
            X,
            y,
            era_id,
            X_eval,
            y_eval,
            eval_every_n_trees,
            early_stopping_rounds,
            eval_metric,
            sample_weight,
            sample_weight_eval,
            eval_aux,
            train_aux,
        )

        if sample_weight is not None:
            sample_weight = self._normalize_sample_weight(sample_weight)
        if X_eval is not None and y_eval is not None and sample_weight_eval is not None:
            sample_weight_eval = self._normalize_sample_weight(sample_weight_eval)

        # Per-model generator (fresh per fit): fit never touches the global RNG state, so
        # concurrent fits in several threads / on several GPUs stay reproducible
        self._rng = torch.Generator(device=self.device)
        if self.random_state is not None:
            self._rng.manual_seed(self.random_state)
        else:
            self._rng.seed()

        continuing = self.warm_start and self._is_fitted

        # ─── Warm Start Logic ───
        if not continuing:
            # Fresh start: reset training state
            self._is_fitted = False
            self._trees_trained = 0
            self.training_loss = []
            self._reset_eval_state()
            self._per_era_importance_all = None
            self._invalidate_prediction_cache()
        else:
            # Continuing training: validate that data is compatible
            if X.shape[1] != self.num_features:
                raise ValueError(
                    f"X has {X.shape[1]} features, but model was trained with {self.num_features} features."
                )
            if self._trees_trained >= self.n_estimators:
                # Already have all requested trees
                if self.verbose:
                    print(f"Model already has {self._trees_trained} trees (n_estimators={self.n_estimators}). No additional training needed.")
                self._rng = None
                return self
            # The best round is tracked per fit (over this fit's evaluations)
            self.best_iteration_ = None
            self.best_score_ = None
        self._rollback = rollback
        self._per_era_importance_best = None
        self.train_predictions_ = None
        self.prebinned_ = self._resolve_prebinned(X, continuing)

        if era_id is None:
            era_id = np.ones(X.shape[0], dtype="int32")

        self._prepare_monotonic_constraints(X.shape[1])

        # ─── Targets ───
        y_prepared, y_eval_prepared = self._prepare_targets(y, y_eval, continuing)
        n_outputs = self.objective_.num_outputs(y_prepared)
        if continuing and self.num_outputs is not None and self.num_outputs != n_outputs:
            raise ValueError(
                "Number of target outputs must match the fitted model when warm_start=True. "
                f"Got {n_outputs} and {self.num_outputs}."
            )
        self.num_outputs = n_outputs
        self._targets_1d = y.ndim == 1

        if not continuing:
            self.forest = (
                [] if self._output_strategy() == PER_OUTPUT
                else [{} for _ in range(self.n_estimators)]
            )

        # ─── Train data preprocessing ───
        self.bin_indices, self.era_indices, self.bin_edges, self.unique_eras = (
            self.preprocess_gpu_data(X, era_id, reuse_edges=continuing)
        )
        self.num_samples, self.num_features = X.shape
        self.num_eras = len(self.unique_eras)
        self.era_indices = self.era_indices.to(dtype=torch.int32)

        # Training reads one feature at a time: feature-major bins [F, N] (coalesced reads)
        self.bins_fm = self.bin_indices.T.contiguous()
        self.bin_indices = None
        self._row_ids = torch.arange(self.num_samples, device=self.device)
        self._ones_i64 = torch.ones(self.num_samples, dtype=torch.int64, device=self.device)

        self.Y_gpu, self.tw = self._build_targets(y_prepared, sample_weight)
        # If every hessian equals the row weight (squared error without missing targets and with
        # the same weight for all outputs), the hessian sums are read from the row-weight
        # histogram instead of being accumulated (K + 1 instead of 2K + 1 channels)
        weight = self.tw.weight
        self._unit_hessian = self.objective_._hessian_is_one() and bool(
            (weight == weight[:, :1]).all()
        )
        self.row_weight_gpu = (weight[:, 0] if self._unit_hessian else self.tw.row_weight).contiguous()
        self._tile_rows_ = self._tile_rows(self.num_eras, self.num_bins + 1)
        self._count_scale = self._device_scale(self.row_weight_gpu)
        self._has_monotone = bool(np.any(self.monotonic_constraints_vector_))
        self._mono_vec_gpu = torch.from_numpy(self.monotonic_constraints_vector_).to(self.device)
        free_bytes, _ = torch.cuda.mem_get_info(torch.device(self.device))
        self._hist_budget = _HIST_MEMORY_FRACTION * free_bytes
        self._tree_graphs = {}

        # ─── Initial scores ───
        if continuing:
            base = self._base_score_tensor()
            if base.numel() != self.num_outputs:
                raise ValueError(
                    "Stored base_prediction is incompatible with the requested number "
                    f"of outputs ({self.num_outputs})."
                )
        else:
            base = torch.as_tensor(
                self.objective_.init_score(self.Y_gpu, self.tw.weight),
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
            if base.numel() == 1 and self.num_outputs > 1:
                base = base.expand(self.num_outputs).clone()
            if base.numel() != self.num_outputs:
                raise ValueError(
                    f"init_score returned {base.numel()} values, expected {self.num_outputs}."
                )
            self._store_base_score(base)

        self.scores = self._predict_raw_binned(self.bins_fm, self._trees_trained, feature_major=True)
        self._best_train_scores = None
        self._train_aux = self._aux_tensor(train_aux)

        self.feature_indices = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

        # ─── Optional Eval Set ───
        if X_eval is not None and y_eval is not None:
            self.bin_indices_eval = self.bin_inference_data(X_eval, name="X_eval")
            self.Y_gpu_eval, self.tw_eval = self._build_targets(
                y_eval_prepared, sample_weight_eval
            )
            self.eval_every_n_trees = eval_every_n_trees
            self.early_stopping_rounds = early_stopping_rounds
            self.eval_metric = self._resolve_eval_metric(eval_metric)
            self.eval_metric_name_ = self._metric_name(self.eval_metric)
            self.eval_metric_higher_is_better_ = self._higher_is_better(self.eval_metric)
            self._eval_aux = self._aux_tensor(eval_aux)
            # Tree sums without the base score, see _add_round_to_eval_scores
            zero = torch.zeros(self.num_outputs, dtype=torch.float32, device=self.device)
            self._eval_acc = self._predict_raw_staged_binned(
                self.bin_indices_eval, [self._trees_trained], base=zero
            )[0]
            self._eval_scores = self._base_score_tensor() + self._eval_acc
        else:
            self.bin_indices_eval = None
            self.Y_gpu_eval = None
            self.tw_eval = None
            self.eval_every_n_trees = None
            self.early_stopping_rounds = None
            self._eval_aux = None
            self._eval_scores = None
            self._eval_acc = None

        # ─── Grow the forest ───
        with torch.no_grad():
            self.grow_forest()

        # In-sample raw scores of the default model (best round when rolled back)
        train_scores = self.scores
        if self._default_num_trees() != self._trees_trained and self._best_train_scores is not None:
            train_scores = self._best_train_scores
        self.train_predictions_ = train_scores.cpu().numpy()

        # Release training buffers
        self.bin_indices = None
        self.bins_fm = None
        self._row_ids = None
        self._ones_i64 = None
        self._tree_graphs = {}
        self._eval_acc = None
        self.Y_gpu = None
        self.tw = None
        self.scores = None
        self.row_weight_gpu = None
        self.bin_indices_eval = None
        self.Y_gpu_eval = None
        self.tw_eval = None
        self._eval_scores = None
        self._eval_aux = None
        self._train_aux = None
        self._best_train_scores = None
        self._rng = None

        return self

    def _prepare_targets(self, y, y_eval, continuing):
        """Encode class labels (classifiers) and convert targets to [N, T] float32."""
        obj = self.objective_
        if obj.is_classifier:
            if not continuing or self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                self.classes_ = self.label_encoder.classes_
                self.num_classes = len(self.classes_)
            else:
                # Warm start: use existing label encoder
                y_encoded = self.label_encoder.transform(y)

            if self.objective == "binary" and self.num_classes != 2:
                raise ValueError(f"binary objective requires exactly 2 classes, got {self.num_classes}")
            if self.num_classes < 2:
                raise ValueError(f"{obj.name} objective requires at least 2 classes, got {self.num_classes}")
            if isinstance(obj, Softmax):
                obj.num_classes = self.num_classes

            y_prepared = obj.prepare_targets(y_encoded)
            y_eval_prepared = (
                obj.prepare_targets(self.label_encoder.transform(y_eval))
                if y_eval is not None
                else None
            )
        else:
            y_prepared = obj.prepare_targets(y)
            y_eval_prepared = obj.prepare_targets(y_eval) if y_eval is not None else None
        return y_prepared, y_eval_prepared

    def _resolve_eval_metric(self, eval_metric):
        obj = self.objective_
        if callable(eval_metric):
            return eval_metric
        if eval_metric is None:
            metric = obj.default_metric
            if metric == "loss" and not obj.has_loss():
                metric = "mse"
            return metric
        if obj.is_classifier and eval_metric not in _CLASSIFICATION_METRICS:
            # Regression metrics are not meaningful for class labels
            return "logloss"
        return eval_metric

    def _resolve_prebinned(self, X, continuing):
        """Whether X holds bin indices; decided once per model and stored as prebinned_."""
        if continuing:
            if self.prebinned is not None and bool(self.prebinned) != self.prebinned_:
                raise ValueError(
                    f"prebinned={self.prebinned} does not match the fitted model "
                    f"(prebinned_={self.prebinned_}) when warm_start=True."
                )
            return self.prebinned_
        if self.prebinned is not None:
            return bool(self.prebinned)

        # Legacy auto-detection (prebinned=None): integers that all lie in [0, num_bins - 1]
        if not np.issubdtype(X.dtype, np.integer) or X.size == 0:
            return False
        if X.min() >= 0 and X.max() < self.num_bins:
            if self.verbose:
                print("Detected pre-binned integer input — skipping quantile binning.")
            return True
        warnings.warn(
            f"Integer X has values outside [0, {self.num_bins - 1}] and is quantile-binned as raw "
            "features. Pass prebinned=True for bin indices (-1 = missing) or prebinned=False to "
            "silence this warning.",
            UserWarning,
            stacklevel=3,
        )
        return False

    @_on_model_device
    def preprocess_gpu_data(self, X_np, era_id_np, reuse_edges=False):
        with torch.no_grad():
            self.num_samples, self.num_features = X_np.shape

            era_id_gpu = torch.from_numpy(era_id_np).to(
                dtype=torch.int32, device=self.device
            )
            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)

            if self.prebinned_:
                # Bin indices are used as given; there are no bin edges
                bin_indices = _prebinned_to_gpu(X_np, self.num_bins, self.device)
                return bin_indices, era_indices, None, unique_eras

            if reuse_edges and self.bin_edges is not None:
                self.bin_edges = self.bin_edges.to(self.device)
                bin_indices = self.bin_inference_data(X_np)
                return bin_indices, era_indices, self.bin_edges, unique_eras

            X_gpu = _to_float_gpu(X_np, self.device)
            # NaNs are excluded from the edges and get their own bin
            bin_edges = _quantile_edges(X_gpu, self.num_bins)
            bin_indices = _bin_matrix(X_gpu, bin_edges, self.nan_bin)
            return bin_indices, era_indices, bin_edges, unique_eras

    # ─── Tree growing ───
    #
    # Trees are grown level by level; all nodes of a level are handled by a few kernel launches,
    # without host syncs:
    #   1. node totals (node_sums) -> leaf values and monotone bounds;
    #   2. histograms [nodes, F, E, B, C] of the level: the smaller child of every split is built
    #      and its sibling is parent - child (exact integer subtraction in deterministic mode);
    #   3. split candidates of all nodes (level_split) and the best split per node (select_split);
    #   4. rows move to their child (heap numbering: the children of node j are 2j and 2j + 1).
    # Nodes are numbered densely per level; nodes below a leaf are inactive and cost nothing. Trees
    # are full-depth with independent node splits, so this grows the same trees as a depth-first
    # recursion. The tree is copied to the host once, at the end.
    #
    # Histogram channels C = Kc + Kh + 1: gradients, hessians (Kh = 0 when every hessian equals
    # the row weight, see fit) and row weights. Blocks accumulate fixed-point integers; with
    # deterministic=True the histograms are int64 (exact, order independent), with
    # deterministic=False float32 (float atomics, half the memory and traffic).

    @staticmethod
    def _tile_rows(num_eras, num_hist_bins):
        """Rows per histogram block; the fixed-point scales are derived from it."""
        rows = 1024
        while rows < 2 * num_eras * num_hist_bins and rows < 8192:
            rows *= 2
        return rows

    def _device_scale(self, values):
        """
        Power-of-two fixed-point scale as a 0-d float32 tensor on the device (no host sync): the
        largest scale for which one block's sum of |values| cannot overflow 32 bits.
        """
        one = torch.ones((), dtype=torch.float64, device=values.device)
        if values.numel() == 0:
            return one.float()
        bound = values.abs().amax().double() * self._tile_rows_
        x = (2.0**30) / bound
        _, exponent = torch.frexp(x)  # x = m * 2^exponent with m in [0.5, 1)
        scale = torch.ldexp(one, (exponent - 1).clamp(-126, 126))
        return torch.where(torch.isfinite(x) & (bound > 0), scale, one).float()

    @staticmethod
    def _acc_to_float(acc, scales, Kc, Kh):
        """Float values of fixed-point sums [..., C] (identity for float sums)."""
        if acc.dtype != torch.int64:
            return acc
        # Powers of two: int64 -> float rounds once, the scaling is exact (as in the split kernel)
        inv = 1.0 / scales
        inv = torch.cat([inv[0].expand(Kc), inv[1].expand(Kh), inv[2:3]])
        return acc.float() * inv

    def _count(self, index, n):
        """Occurrences of 0 … n-1 in index [N] (bincount without its host sync)."""
        counts = torch.zeros(n, dtype=torch.int64, device=self.device)
        return counts.index_add_(0, index.long(), self._ones_i64[: index.numel()])

    def _level_bytes(self, num_features, num_channels):
        """(histogram bytes, split buffer bytes) per node."""
        E, B, n_cand = self.num_eras, self.num_bins + 1, 2 * self.num_bins
        elem = 8 if self.deterministic else 4
        hist = num_features * E * B * num_channels * elem
        split = num_features * n_cand * (4 * E + 9)
        return hist, split

    def _build_histograms(self, node, build, slot_of_node, max_rows, out, grad, hess_ch, scales):
        """Histograms of the nodes with build[j] (rows with node_of_row == j) into out[slot_of_node[j]]."""
        n_nodes = build.numel()
        T = self._tile_rows_
        key = torch.where(build[node.long()], node, n_nodes)
        order = torch.argsort(key, stable=True).to(torch.int32)
        seg_len = self._count(key, n_nodes + 1)[:n_nodes]
        seg_start = torch.cumsum(seg_len, 0) - seg_len
        tile_offset = torch.zeros(n_nodes + 1, dtype=torch.int64, device=self.device)
        tile_offset[1:] = torch.cumsum((seg_len + T - 1) // T, 0)
        _get_node_kernel().histogram_level(
            self.bins_fm,
            grad,
            hess_ch,
            self.row_weight_gpu,
            order,
            seg_start.to(torch.int32),
            seg_len.to(torch.int32),
            slot_of_node.to(torch.int32),
            tile_offset.to(torch.int32),
            self.feat_indices_tree,
            self.era_indices,
            out,
            scales,
            (max_rows + T - 1) // T + n_nodes,
            T,
        )

    def _split_and_select(self, hist, active_u8, scales, mono_req, Kc, best, best_gain, best_era):
        """Best split of every active node of hist [S, F, E, B, C] into best / best_gain / best_era."""
        S, F = hist.shape[:2]
        n_cand = 2 * self.num_bins
        kernel = _get_node_kernel()
        era_gain = torch.empty((S, F, self.num_eras, n_cand), device=self.device)
        gain = torch.empty((S, F, n_cand), device=self.device)
        agree = torch.empty_like(gain)
        mono_ok = None
        if mono_req is not None:
            mono_ok = torch.empty((S, F, n_cand), dtype=torch.uint8, device=self.device)
        kernel.level_split(
            hist,
            active_u8,
            scales,
            mono_req,
            float(self.min_child_weight),
            float(self.min_sum_hessian),
            float(self.L2_reg),
            Kc,
            era_gain,
            gain,
            agree,
            mono_ok,
        )
        kernel.select_split(
            gain, agree, mono_ok, era_gain, active_u8, float(self.min_split_gain),
            best, best_gain, best_era,
        )

    def _level_splits(self, d, node, active, counts, prev, grad, hess_ch, scales, mono_req):
        """
        Best split per node of level d. prev = [histograms of level d - 1 or None]; the list holds
        the only reference, so they are released as soon as possible. Returns (best [n],
        best_gain [n], best_era [n, E], hist), hist being the level's histograms if they fit
        into the memory budget (else None).
        """
        hist_prev = prev.pop()
        n_d = 1 << d
        dev = self.device
        Kc, C = grad.shape[1], grad.shape[1] + hess_ch.shape[1] + 1
        F = self.feat_indices_tree.numel()
        shape = (F, self.num_eras, self.num_bins + 1, C)
        dtype = torch.int64 if self.deterministic else torch.float32
        hist_bytes, split_bytes = self._level_bytes(F, C)
        best = torch.empty(n_d, dtype=torch.int32, device=dev)
        best_gain = torch.empty(n_d, dtype=torch.float32, device=dev)
        best_era = torch.empty((n_d, self.num_eras), dtype=torch.float32, device=dev)
        active_u8 = active.to(torch.uint8)
        slots = torch.arange(n_d, dtype=torch.int32, device=dev)

        n_prev = n_d // 2 if hist_prev is not None else 0
        if (n_prev + n_d) * hist_bytes + n_d * split_bytes <= self._hist_budget:
            if hist_prev is None:
                hist = torch.zeros((n_d,) + shape, dtype=dtype, device=dev)
                self._build_histograms(node, active, slots, self.num_samples, hist, grad, hess_ch, scales)
            else:
                # Build the smaller child of every split; its sibling is parent - child. Slots of
                # inactive nodes are never read, so only the built slots are zeroed.
                pairs = counts.view(-1, 2)
                left_smaller = (pairs[:, 0] <= pairs[:, 1]).contiguous()
                build = torch.stack([left_smaller, ~left_smaller], dim=1).reshape(-1) & active
                hist = torch.empty((n_d,) + shape, dtype=dtype, device=dev)
                built = 2 * torch.arange(n_d // 2, device=dev) + (~left_smaller).long()
                hist.index_fill_(0, built, 0)
                self._build_histograms(
                    node, build, slots, self.num_samples // 2, hist, grad, hess_ch, scales
                )
                split = active.view(-1, 2)[:, 0].contiguous()
                _get_node_kernel().sibling_subtract(hist, hist_prev, left_smaller, split)
            del hist_prev
            self._split_and_select(hist, active_u8, scales, mono_req, Kc, best, best_gain, best_era)
            return best, best_gain, best_era, hist

        # Too large to keep a whole level: chunks of nodes, each child built directly
        del hist_prev
        chunk = max(1, int(self._hist_budget // (hist_bytes + split_bytes)))
        for a in range(0, n_d, chunk):
            b = min(n_d, a + chunk)
            in_chunk = active & (slots >= a) & (slots < b)
            hist = torch.zeros((b - a,) + shape, dtype=dtype, device=dev)
            self._build_histograms(
                node, in_chunk, (slots - a).clamp(min=0), self.num_samples, hist, grad, hess_ch, scales
            )
            self._split_and_select(
                hist, active_u8[a:b], scales, mono_req, Kc, best[a:b], best_gain[a:b], best_era[a:b]
            )
            del hist
        return best, best_gain, best_era, None

    @staticmethod
    def _child_bounds(raw_val, lower, upper, constraint):
        """
        Monotone value bounds [2n, Kc] of the children of n parents (bounds [n, Kc], constraint [n]
        of the split feature): mid = mean of the children's clamped values bounds both sides.
        """
        n = lower.shape[0]
        children = raw_val.view(n, 2, -1)
        left_val = torch.clamp(children[:, 0], min=lower, max=upper)
        right_val = torch.clamp(children[:, 1], min=lower, max=upper)
        mid = 0.5 * (left_val + right_val)
        inc = (constraint > 0).view(-1, 1)
        dec = (constraint < 0).view(-1, 1)
        left_lower = torch.where(dec, torch.maximum(lower, mid), lower)
        left_upper = torch.where(inc, torch.minimum(upper, mid), upper)
        right_lower = torch.where(inc, torch.maximum(lower, mid), lower)
        right_upper = torch.where(dec, torch.minimum(upper, mid), upper)
        new_lower = torch.stack([left_lower, right_lower], dim=1).reshape(2 * n, -1)
        new_upper = torch.stack([left_upper, right_upper], dim=1).reshape(2 * n, -1)
        return new_lower, new_upper

    def _run_tree(self, grad, hess):
        """
        (score update [N, Kc], records) of one tree, see _grow_tree_device.

        Every tree of a fit launches the same kernels on tensors of the same shapes, without host
        syncs. The second tree of each shape is therefore captured as a CUDA graph, and later
        trees replay it: one launch instead of ~100 small ones per tree. Inputs that change
        between trees (gradients, hessians, sampled features) are copied into the graph's static
        buffers. The first tree runs eagerly (warms up the kernels and the caching allocator).
        """
        key = grad.shape[1]
        state = self._tree_graphs.get(key) if self._use_cuda_graphs else "eager"
        if state is None or state == "eager":
            if state is None:
                self._tree_graphs[key] = "capture"
            return self._grow_tree_device(grad, hess)
        with torch.cuda.device(self.device):
            if state == "capture":
                static = (grad.clone(), hess.clone(), self.feat_indices_tree.clone())
                graph = torch.cuda.CUDAGraph()
                sampled = self.feat_indices_tree
                self.feat_indices_tree = static[2]
                try:
                    with _GRAPH_CAPTURE_LOCK, torch.cuda.graph(
                        graph, stream=_capture_stream(), capture_error_mode=_CAPTURE_MODE
                    ):
                        outputs = self._grow_tree_device(static[0], static[1])
                except RuntimeError:
                    # Capture is an optimization only: fall back to eager trees
                    self._tree_graphs[key] = "eager"
                    return self._grow_tree_device(grad, hess)
                finally:
                    self.feat_indices_tree = sampled
                state = self._tree_graphs[key] = (graph, static, outputs)
            else:
                graph, static, outputs = state
                static[0].copy_(grad)
                static[1].copy_(hess)
                static[2].copy_(self.feat_indices_tree)
            state[0].replay()
            return state[2]

    def _grow_tree_device(self, grad, hess):
        """
        Grow one tree on the GPU, without host syncs. Returns the training score update
        [N, Kc] and the tree as a float64 record tensor (see _tree_from_records).
        """
        kernel = _get_node_kernel()
        dev = self.device
        N = self.num_samples
        Kc = grad.shape[1]
        hess_ch = hess[:, :0] if self._unit_hessian else hess
        Kh = hess_ch.shape[1]
        C = Kc + Kh + 1
        D = self.max_depth
        Bv = self.num_bins
        n_cand = 2 * Bv
        feats = self.feat_indices_tree.long()
        mono = self._has_monotone
        mono_req = (
            self.monotonic_required_split_direction_gpu_[feats].contiguous() if mono else None
        )
        # Fixed-point scales of the shared-memory accumulation (both modes, see histogram_kernel.cu)
        scales = torch.stack(
            [self._device_scale(grad), self._device_scale(hess_ch), self._count_scale]
        )
        # Checked on the host with the tree, instead of a sync per round
        finite = torch.isfinite(grad).all() & torch.isfinite(hess).all()

        node = torch.zeros(N, dtype=torch.int32, device=dev)  # node of every row at level d
        active = torch.ones(1, dtype=torch.bool, device=dev)
        hist = None
        lower = upper = constraint = None
        leaf_val = None  # value of the leaf a node lies in (for nodes below a leaf)
        records = []
        for d in range(D + 1):
            n_d = 1 << d
            # 1. Node totals -> values (clamped to the monotone bounds)
            sums = torch.zeros(
                (n_d, C), dtype=torch.int64 if self.deterministic else torch.float32, device=dev
            )
            # Blocks of at most tile_rows rows keep the fixed-point sums within 32 bits
            kernel.node_sums(
                node, grad, hess_ch, self.row_weight_gpu, sums, scales, min(256, self._tile_rows_)
            )
            sums = self._acc_to_float(sums, scales, Kc, Kh)
            G = sums[:, :Kc]
            H = sums[:, Kc : Kc + Kh] if Kh else sums[:, -1:].expand(-1, Kc)
            val = G / (H + self.L2_reg)
            if mono:
                if d == 0:
                    lower = torch.full((1, Kc), float("-inf"), device=dev)
                    upper = torch.full((1, Kc), float("inf"), device=dev)
                else:
                    lower, upper = self._child_bounds(val, lower, upper, constraint)
                val = torch.clamp(val, min=lower, max=upper)
            counts = self._count(node, n_d)

            # 2./3. Best split of every active node
            if d < D:
                prev, hist = [hist], None
                best, best_gain, best_era, hist = self._level_splits(
                    d, node, active, counts, prev, grad, hess_ch, scales, mono_req
                )
                valid = best >= 0
            else:
                best = torch.full((n_d,), -1, dtype=torch.int32, device=dev)
                best_gain = torch.zeros(n_d, device=dev)
                best_era = torch.zeros((n_d, self.num_eras), device=dev)
                valid = torch.zeros(n_d, dtype=torch.bool, device=dev)
            leaf = active & ~valid
            if leaf_val is None:
                leaf_val = val
            else:
                leaf_val = torch.where(leaf[:, None], val, leaf_val.repeat_interleave(2, dim=0))

            candidate = best.clamp(min=0).long()
            f_global = feats[candidate // n_cand]
            candidate = candidate % n_cand
            records.append(
                torch.cat(
                    [
                        torch.stack(
                            [t.double() for t in (active, valid, f_global, candidate, best_gain, counts)],
                            dim=1,
                        ),
                        best_era.double(),
                        val.double(),
                    ],
                    dim=1,
                ).reshape(-1)
            )
            if d == D:
                break

            # 4. Rows move to their child; rows of leaves to the (inactive) left child
            nl = node.long()
            row_bins = self.bins_fm.view(-1)[f_global[nl] * N + self._row_ids]
            split_bin = (candidate % Bv)[nl]
            nan_left = (candidate >= Bv)[nl]
            go_left = (row_bins <= split_bin) | (nan_left & (row_bins == self.nan_bin))
            node = node * 2 + (valid[nl] & ~go_left).to(torch.int32)
            if mono:
                constraint = torch.where(valid, self._mono_vec_gpu[f_global], 0)
            active = valid.repeat_interleave(2)

        # Training scores: every row gets the value of its leaf
        update = leaf_val[node.long()] * self.learning_rate
        records.append(finite.double().view(1))
        return update, torch.cat(records)

    def _tree_from_records(self, host, Kc):
        """
        Tree dict from the records of _grow_tree_device: per level d, [2^d, 6 + E + Kc] rows
        (active, valid split, feature, candidate, gain, rows, per-era gains, node values),
        then the finiteness flag of the gradients.
        """
        if not host[-1]:
            raise FloatingPointError(
                f"Objective {self.objective_!r} produced non-finite gradients or hessians."
            )
        D = self.max_depth
        Bv = self.num_bins
        E = self.num_eras
        width = 6 + E + Kc
        levels = []
        offset = 0
        for d in range(D + 1):
            levels.append(host[offset : offset + (width << d)].reshape(1 << d, width))
            offset += width << d
        importance = self._feature_importance_acc

        def build(d, j):
            r = levels[d][j]
            if r[1]:
                feature, cand = int(r[2]), int(r[3])
                # Pre-order, like the order in which splits are made
                importance[:, feature] += r[6 : 6 + E].astype(np.float32)
                node = {"feature": feature, "bin": cand % Bv, "nan_left": cand >= Bv}
                node["left"] = build(d + 1, 2 * j)
                node["right"] = build(d + 1, 2 * j + 1)
                node["gain"] = float(r[4])
                return node
            value = r[6 + E :]
            return {
                "leaf_value": float(value[0]) if Kc == 1 else [float(v) for v in value],
                "samples": int(r[5]),
            }

        return build(0, 0)

    def grow_forest(self):
        """Gradient boosting loop shared by all objectives."""
        # Per-era importance of the trees from earlier (warm-start) fits
        shape = (self.num_eras, self.num_features)
        base = np.zeros(shape, dtype=np.float32)
        previous = self._per_era_importance_all
        # Unknown for earlier trees (e.g. after truncate() to a round without a snapshot)
        base_known = previous is not None or self._trees_trained == 0
        if previous is not None and previous.shape == shape:
            base += previous
        elif previous is not None:
            # Warm start with a different era layout: keep the totals in a single row
            base[0] = previous.sum(axis=0)

        self.stop = False
        self._evals_since_best = 0
        self._best_importance = None
        strategy = self._output_strategy()

        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            k = self.num_features

        # Per-era split gains of this fit, accumulated on the host in split order (deterministic)
        self._feature_importance_acc = np.zeros((self.num_eras, self.num_features), dtype=np.float32)

        # Warm start: start from where we left off
        start_iter = self._trees_trained if self.warm_start and self._is_fitted else 0

        if strategy == PER_OUTPUT:
            del self.forest[start_iter:]
        elif len(self.forest) < self.n_estimators:
            # Ensure forest has enough slots
            self.forest.extend([{} for _ in range(self.n_estimators - len(self.forest))])

        # Software pipeline: round i + 1 is enqueued on the GPU before the host turns round i
        # into tree dicts and evaluates it, so host work overlaps with GPU work. When early
        # stopping ends training at round i, the speculative round i + 1 is dropped; its score
        # update does not matter then (training stops with a rollback to the best round).
        pending = None
        for i in range(start_iter, self.n_estimators):
            # Finiteness is checked with the tree's host transfer (see _tree_from_records)
            g, h = compute_grad_hess(
                self.objective_, self.Y_gpu, self.scores, self.tw.weight, self.tw.mask,
                check_finite=False,
            )
            # Trees are grown on the negative gradient: leaf = sum(-g) / (sum(h) + L2)
            neg_grad = -g

            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(
                    self.num_features, device=self.device, dtype=torch.int32, generator=self._rng
                )[:k]

            if strategy == PER_OUTPUT:
                launched = [
                    self._launch_tree(
                        neg_grad[:, col : col + 1].contiguous(),
                        h[:, col : col + 1].contiguous(),
                        score_col=col,
                    )
                    for col in range(self.num_outputs)
                ]
            else:
                launched = [self._launch_tree(neg_grad, h)]
            # Training scores after round i, for its evaluation (self.scores moves on)
            round_scores = self.scores.clone() if self.eval_every_n_trees is not None else None

            if pending is not None:
                self._finish_round(*pending)
                pending = None
                if self.stop:
                    break
            pending = (i, launched, round_scores)
        if pending is not None:
            self._finish_round(*pending)

        # Per-era importance of all trees, and of the best round of this fit
        if base_known:
            self._per_era_importance_all = base + self._feature_importance_acc
            if self._best_importance is not None:
                self._per_era_importance_best = base + self._best_importance
        self._feature_importance_acc = None
        self._best_importance = None
        self._is_fitted = True
        self._update_importance_attributes()

        if self.verbose:
            if strategy == PER_OUTPUT:
                print(f"Finished training {self.objective_.name} forest. Total rounds: {self._trees_trained} ({self._trees_trained * self.num_outputs} trees)")
            else:
                print(f"Finished training {self.objective_.name} forest. Total trees: {self._trees_trained}")

    def _launch_tree(self, grad, hess, score_col=None):
        """Enqueue one tree: updates the training scores, starts the copy of its records."""
        update, records = self._run_tree(grad, hess)
        if score_col is not None:
            self.scores[:, score_col] += update[:, 0]
        else:
            self.scores += update
        host = torch.empty(records.shape, dtype=records.dtype, pin_memory=True)
        host.copy_(records, non_blocking=True)
        # Recorded on the stream of the copy (the model device's, not the caller's current one)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(records.device))
        return host, done, grad.shape[1]

    def _finish_round(self, i, launched, round_scores):
        """Turn the records of round i into tree dicts, store and evaluate the round."""
        trees = []
        for host, done, Kc in launched:
            done.synchronize()
            trees.append(self._tree_from_records(host.numpy(), Kc))
        if self._output_strategy() == PER_OUTPUT:
            self.forest.append(trees)
        else:
            self.forest[i] = trees[0]
        self._trees_trained = i + 1
        self.compute_eval(i, round_scores)

    # ─── Evaluation ───

    @staticmethod
    def _metric_name(metric):
        if isinstance(metric, Metric):
            return metric.name
        return getattr(metric, "__name__", str(metric))

    @staticmethod
    def _higher_is_better(metric):
        if isinstance(metric, Metric):
            return metric.higher_is_better
        return isinstance(metric, str) and metric in _HIGHER_IS_BETTER

    def _aux_tensor(self, aux):
        if aux is None:
            return None
        return torch.from_numpy(np.ascontiguousarray(aux, dtype=np.float32)).to(self.device)

    def _metric_value(self, metric, Y, raw, tw, aux=None):
        """Evaluate a metric on raw scores raw [N, K]; returns its natural value."""
        obj = self.objective_
        if metric == "loss":
            return weighted_loss(obj, Y, raw, tw.weight, tw.mask)

        pred = obj.transform(raw)
        target_weight = tw.weight if tw.weight.shape == tw.target_mask.shape else tw.row_weight[:, None]

        if callable(metric):
            y_true, mask, weight = Y, tw.target_mask, target_weight
            if obj.is_classifier:
                y_true, mask, weight = Y[:, 0], mask[:, 0], weight[:, 0]
            elif self._targets_1d and pred.shape[1] == 1:
                pred, y_true, mask, weight = pred[:, 0], Y[:, 0], mask[:, 0], weight[:, 0]
            if aux is not None:
                return float(metric(y_true, pred, weight, mask, aux=aux))
            return float(metric(y_true, pred, weight, mask))

        if metric == "logloss":
            return weighted_log_loss_torch(Y[:, 0], pred, sample_weight=tw.row_weight).item()
        if metric == "accuracy":
            labels = torch.argmax(pred, dim=1)
            return weighted_accuracy_torch(Y[:, 0], labels, sample_weight=tw.row_weight).item()
        if metric == "mse":
            return weighted_multi_mse_torch(
                Y, pred, sample_weight=target_weight, valid_mask=tw.target_mask
            ).item()

        valid = tw.target_mask[:, 0]
        y_v, p_v, w_v = Y[valid, 0], pred[valid, 0], target_weight[valid, 0]
        if metric == "corr":
            return 1.0 - weighted_corr_loss_torch(y_v, p_v, sample_weight=w_v).item()
        if metric == "rmsle":
            return weighted_rmsle_torch(y_v, p_v, sample_weight=w_v).item()
        raise ValueError(f"Invalid eval_metric: {metric}.")

    def _is_better(self, score, best):
        if not math.isfinite(score):
            return False
        if best is None:
            return True
        return score > best if self.eval_metric_higher_is_better_ else score < best

    def _add_round_to_eval_scores(self, i):
        """
        Add round i to the eval scores. The tree sums are accumulated with the same additions as
        the prediction kernel, so the eval scores equal predict_raw(X_eval, num_trees=i + 1).
        """
        self._ensure_packed(i + 1)
        nodes, leaf, out_col = (self._to_device(a) for a in self._pack_rounds[i])
        if self._output_strategy() != PER_OUTPUT:
            out_col = None
        self._eval_acc = self._forest_kernel(
            self.bin_indices_eval, nodes, leaf, out_col, self._eval_acc, [nodes.size(0)]
        )[0]
        self._eval_scores = self._base_score_tensor() + self._eval_acc

    @_on_model_device
    def compute_eval(self, i, train_scores=None):
        """Evaluate round i; train_scores are the training scores after round i (default: current)."""
        if self.eval_every_n_trees is None:
            return
        if train_scores is None:
            train_scores = self.scores

        self._add_round_to_eval_scores(i)

        obj = self.objective_
        loss_name = "loss" if obj.has_loss() else "mse"
        # The training loss costs a full pass over the training set and a host sync, so by
        # default it is only computed at evaluation checkpoints
        train_loss = None
        if self.record_training_loss == "all":
            train_loss = self._metric_value(loss_name, self.Y_gpu, train_scores, self.tw)
            self.training_loss.append(train_loss)

        # Evaluate every eval_every_n_trees rounds and always at the last round, so that the
        # trees after the last regular checkpoint are not dropped by the rollback
        if i % self.eval_every_n_trees != 0 and i != self.n_estimators - 1:
            return

        metric = self.eval_metric
        eval_score = self._metric_value(
            metric, self.Y_gpu_eval, self._eval_scores, self.tw_eval, self._eval_aux
        )
        # The eval metric on the training set (skipped if it needs aux data we don't have)
        train_score = None
        if self._train_aux is not None or not (
            callable(metric) and self._aux_mode(metric) == "required"
        ):
            train_score = self._metric_value(metric, self.Y_gpu, train_scores, self.tw, self._train_aux)
            self.evals_result_["train"].append(train_score)
        self.evals_result_["eval"].append(eval_score)
        self.eval_iterations_.append(i + 1)

        if self.record_training_loss == "eval":
            if isinstance(metric, str) and metric == loss_name and train_score is not None:
                train_loss = train_score
            else:
                train_loss = self._metric_value(loss_name, self.Y_gpu, train_scores, self.tw)
            self.training_loss.append(train_loss)

        if self._is_better(eval_score, self.best_score_):
            self.best_score_ = eval_score
            self.best_iteration_ = i + 1
            self._evals_since_best = 0
            self._best_importance = self._feature_importance_acc.copy()
            if self._rollback:
                self._best_train_scores = train_scores.clone()
        else:
            self._evals_since_best += 1
            if self._evals_since_best >= self.early_stopping_rounds:
                self.stop = True

        if not self.verbose:
            return
        name = self.eval_metric_name_
        if train_score is not None:
            train_part = f"Train {name}: {train_score:.6f} | "
        elif train_loss is not None:
            train_part = f"Train {loss_name}: {train_loss:.6f} | "
        else:
            train_part = ""
        print(f"🌲 Tree {i+1}/{self.n_estimators} | {train_part}Eval {name}: {eval_score:.6f}")

    # ─── Binning / prediction ───

    @_on_model_device
    def bin_data_with_existing_edges(self, X_np):
        """Internal int8 bins [N, F] of raw features X_np with the fitted bin edges."""
        if self.bin_edges is None:
            raise ValueError("This model has no bin edges (it was fitted on pre-binned data).")
        with torch.no_grad():
            X_gpu = _to_float_gpu(X_np, self.device)
            return _bin_matrix(X_gpu, self.bin_edges.to(self.device), self.nan_bin)

    def _check_input_shape(self, X_np, name):
        if X_np.ndim != 2 or X_np.shape[1] != self.num_features:
            raise ValueError(
                f"{name} must have shape (n_samples, {self.num_features}), got {X_np.shape}."
            )

    @_on_model_device
    def bin_inference_data(self, X_np, name="X"):
        """
        Internal int8 bins [N, F] on the model device: raw features are binned with the fitted
        edges; for a pre-binned model (prebinned_), X_np must hold bin indices (-1 = missing).
        """
        self._check_input_shape(X_np, name)
        if self.prebinned_:
            return _prebinned_to_gpu(X_np, self.num_bins, self.device, name)
        return self.bin_data_with_existing_edges(X_np)

    def _bin_inference_cpu(self, X_np, name="X"):
        """bin_inference_data on the CPU: internal int8 bins [N, F] as a numpy array."""
        self._check_input_shape(X_np, name)
        if self.prebinned_:
            return _prebinned_to_cpu(X_np, self.num_bins, name)
        if self.bin_edges is None:
            raise ValueError("This model has no bin edges (it was fitted on pre-binned data).")
        if self._edges_cpu is None:
            self._edges_cpu = self.bin_edges.detach().cpu().numpy().astype(np.float32, copy=False)
        if torch.is_tensor(X_np):
            X_np = X_np.detach().cpu().numpy()
        return _bin_matrix_cpu(X_np, self._edges_cpu, self.nan_bin)

    def _base_score_np(self):
        """Initial raw score as a [K] float32 numpy array (cached)."""
        if self._base_cpu is None:
            base = self.base_prediction
            K = self.num_outputs
            if base is None:
                base = np.zeros(K, dtype=np.float32)
            elif torch.is_tensor(base):
                base = base.detach().cpu().numpy()
            base = np.asarray(base, dtype=np.float32).reshape(-1)
            if base.size == 1 and K > 1:
                base = np.repeat(base, K)
            self._base_cpu = base
        return self._base_cpu

    @staticmethod
    def _use_cpu(device):
        """Prediction device: None -> the model's GPU, "cpu" -> the NumPy path."""
        if device is None:
            return False
        try:
            device_type = torch.device(device).type
        except (RuntimeError, TypeError) as e:
            raise ValueError(f"device must be None or 'cpu', got {device!r}.") from e
        if device_type != "cpu":
            raise ValueError(
                f"device must be None (the model device) or 'cpu', got {device!r}."
            )
        return True

    def _predict_raw_binned(self, bin_indices, num_trees=None, feature_major=False):
        """Raw scores [N, K] for binned data, using the first num_trees rounds (None: default model)."""
        num_trees = self._resolve_num_trees(num_trees)
        return self._predict_raw_staged_binned(bin_indices, [num_trees], feature_major)[0]

    def _staged_raw(self, X_np, rounds, device):
        """Raw scores [P, N, K] (torch, on the GPU or the CPU) after each of rounds (ascending)."""
        if self._use_cpu(device):
            return torch.from_numpy(
                self._predict_raw_staged_cpu(self._bin_inference_cpu(X_np), rounds)
            )
        return self._staged_raw_gpu(X_np, rounds)

    @_on_model_device
    def _staged_raw_gpu(self, X_np, rounds):
        bin_indices = self.bin_inference_data(X_np)
        return self._predict_raw_staged_binned(bin_indices, rounds)

    def _resolve_rounds(self, rounds):
        """Validated round counts and the sorted unique ones: (rounds, unique, positions)."""
        if isinstance(rounds, (int, np.integer)) and not isinstance(rounds, bool):
            raise TypeError("rounds must be a sequence of round counts, e.g. [10, 20, 50].")
        rounds = [self._resolve_num_trees(r) for r in rounds]
        if not rounds:
            raise ValueError("rounds must not be empty.")
        unique = sorted(set(rounds))
        positions = [unique.index(r) for r in rounds]
        return unique, positions

    def _output_shape(self, raw):
        """Drop the output axis of single-output regression-like models ([..., N, 1] -> [..., N])."""
        if not self.objective_.is_classifier and self._targets_1d and raw.shape[-1] == 1:
            return raw[..., 0]
        return raw

    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

    @_on_model_device
    def predict_binned(self, bin_indices, num_trees=None):
        """Transformed predictions for binned data ([N] for single-output targets)."""
        pred = self.objective_.transform(self._predict_raw_binned(bin_indices, num_trees))
        return self._output_shape(pred)

    def predict_raw(self, X_np, num_trees=None, device=None):
        """
        Raw (untransformed) scores, e.g. logits for classification.

        num_trees: number of boosting rounds to use. None uses the default model, i.e.
        best_iteration_ after early stopping and all trained rounds otherwise.
        device: None predicts on the model's GPU; "cpu" uses a NumPy implementation without any
        CUDA call (lowest latency for a few rows). Both give identical raw scores (transforms
        such as softmax may differ in the last bit).

        Returns shape (n_samples,) for single-output regression-like models, else (n_samples, K).
        """
        self._check_fitted()
        num_trees = self._resolve_num_trees(num_trees)
        raw = self._staged_raw(X_np, [num_trees], device)[0]
        return self._output_shape(raw).cpu().numpy()

    def predict(self, X_np, num_trees=None, device=None):
        """
        Predict on new data.

        For regression / custom objectives: returns objective.transform(raw scores)
        For classification: returns predicted class labels

        num_trees: number of boosting rounds to use. None uses the default model, i.e.
        best_iteration_ after early stopping and all trained rounds otherwise.
        device: None (model GPU) or "cpu", see predict_raw.
        """
        self._check_fitted()
        num_trees = self._resolve_num_trees(num_trees)
        if self.objective_.is_classifier:
            # Classification: return class labels
            probs = self.predict_proba(X_np, num_trees, device=device)
            class_indices = np.argmax(probs, axis=1)
            return self.label_encoder.inverse_transform(class_indices)

        raw = self._staged_raw(X_np, [num_trees], device)[0]
        return self._output_shape(self.objective_.transform(raw)).cpu().numpy()

    def predict_proba(self, X_np, num_trees=None, device=None):
        """
        Predict class probabilities (classification only).

        num_trees: number of boosting rounds to use (None: default model, see predict).
        device: None (model GPU) or "cpu", see predict_raw.

        Returns:
            np.array of shape (n_samples, n_classes) with probabilities
        """
        if not self._is_classifier():
            raise ValueError("predict_proba only available for classification objectives")
        self._check_fitted()
        num_trees = self._resolve_num_trees(num_trees)
        raw = self._staged_raw(X_np, [num_trees], device)[0]
        return self.objective_.transform(raw).cpu().numpy()

    @_on_model_device
    def predict_proba_binned(self, bin_indices, num_trees=None):
        """
        Predict probabilities from pre-binned data (classification).

        Returns:
            torch.Tensor of shape (n_samples, n_classes) with probabilities
        """
        return self.objective_.transform(self._predict_raw_binned(bin_indices, num_trees))

    def staged_predict_raw(self, X_np, rounds, device=None):
        """
        Raw scores after several numbers of boosting rounds, from one pass over the forest.

        rounds: sequence of round counts (0 … num_trees_), any order, duplicates allowed.
        device: None (model GPU) or "cpu", see predict_raw.

        Returns an array of shape (len(rounds), n_samples) for single-output regression-like
        models, else (len(rounds), n_samples, K); entry p equals
        predict_raw(X_np, num_trees=rounds[p]) exactly.
        """
        self._check_fitted()
        unique, positions = self._resolve_rounds(rounds)
        raw = self._staged_raw(X_np, unique, device)[positions]
        return self._output_shape(raw).cpu().numpy()

    def staged_predict(self, X_np, rounds, device=None):
        """
        predict() after several numbers of boosting rounds, from one pass over the forest.

        Returns an array with a leading axis of length len(rounds): transformed predictions (class
        labels for classifiers); entry p equals predict(X_np, num_trees=rounds[p]) exactly.
        """
        self._check_fitted()
        unique, positions = self._resolve_rounds(rounds)
        raw = self._staged_raw(X_np, unique, device)
        pred = torch.stack([self.objective_.transform(r) for r in raw])[positions]
        if self.objective_.is_classifier:
            labels = torch.argmax(pred, dim=2).cpu().numpy()
            return np.stack([self.label_encoder.inverse_transform(l) for l in labels])
        return self._output_shape(pred).cpu().numpy()

    def staged_predict_proba(self, X_np, rounds, device=None):
        """predict_proba() after several numbers of boosting rounds: (len(rounds), n_samples, K)."""
        if not self._is_classifier():
            raise ValueError("staged_predict_proba only available for classification objectives")
        self._check_fitted()
        unique, positions = self._resolve_rounds(rounds)
        raw = self._staged_raw(X_np, unique, device)
        return torch.stack([self.objective_.transform(r) for r in raw])[positions].cpu().numpy()

    # ─── Feature importance ───

    def _split_nodes(self, num_trees):
        """Yield every split node of the first num_trees rounds."""
        for round_trees in self.forest[:num_trees]:
            trees = round_trees if isinstance(round_trees, list) else [round_trees]
            for tree in trees:
                stack = [tree] if tree else []
                while stack:
                    node = stack.pop()
                    if "leaf_value" in node:
                        continue
                    yield node
                    stack.extend((node["left"], node["right"]))

    def _total_importance(self, n):
        """Summed (over eras) gain per feature of the first n rounds, unnormalized."""
        totals = np.zeros(self.num_features, dtype=np.float64)
        for node in self._split_nodes(n):
            if "gain" not in node:
                # Trees saved before split gains were stored: only the snapshots are known
                era_importance = self._snapshot_importance(n)
                if era_importance is None:
                    raise ValueError(
                        "This model was saved without per-split gains; feature importance is "
                        "only available for all trained rounds."
                    )
                return era_importance.sum(axis=0)
            totals[node["feature"]] += node["gain"]
        return totals.astype(np.float32)

    def _snapshot_importance(self, n):
        """Per-era importance [E, F] for n rounds if a snapshot exists, else None."""
        if n == self._trees_trained and self._per_era_importance_all is not None:
            return self._per_era_importance_all
        if n == self.best_iteration_ and self._per_era_importance_best is not None:
            return self._per_era_importance_best
        return None

    def _per_era_importance(self, n):
        era_importance = self._snapshot_importance(n)
        if era_importance is not None:
            return era_importance
        if self._per_era_importance_all is not None and self._per_era_importance_all.shape[0] == 1:
            return self._total_importance(n)[None, :]
        raise ValueError(
            "Per-era feature importance is only stored for all trained rounds "
            f"({self._trees_trained}) and for best_iteration_ ({self.best_iteration_}); got num_trees={n}."
        )

    def _update_importance_attributes(self):
        n = self._default_num_trees()
        self.feature_importance_ = self._total_importance(n)
        try:
            self.per_era_feature_importance_ = self._per_era_importance(n)
        except ValueError:
            self.per_era_feature_importance_ = None

    def get_feature_importance(self, importance_type='gain', normalize=True, num_trees=None):
        """
        Get feature importance scores.

        Parameters:
        -----------
        importance_type : str, default='gain'
            Type of importance. Currently only 'gain' is supported.
        normalize : bool, default=True
            If True, normalize importances to sum to 1.0
        num_trees : int or None, default=None
            Count the splits of the first num_trees rounds. None uses the default model
            (best_iteration_ after early stopping, else all trained rounds).

        Returns:
        --------
        np.ndarray
            Array of shape (n_features,) with importance scores.

        Note:
        -----
        When era_id is used during training, this returns the sum of
        per-era importances. Use get_per_era_feature_importance() to
        see era-specific importances.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")

        if importance_type != 'gain':
            raise ValueError(f"importance_type '{importance_type}' not supported. Use 'gain'.")

        importance = self._total_importance(self._resolve_num_trees(num_trees)).copy()

        if normalize and importance.sum() > 0:
            importance = importance / importance.sum()

        return importance

    def get_per_era_feature_importance(self, normalize=True, num_trees=None):
        """
        Get per-era feature importance scores.

        Parameters:
        -----------
        normalize : bool, default=True
            If True, normalize importances within each era to sum to 1.0
        num_trees : int or None, default=None
            None uses the default model (best_iteration_ after early stopping, else all
            trained rounds). Per-era importances are stored for all trained rounds and for
            best_iteration_ only; other values raise (except with a single era).

        Returns:
        --------
        np.ndarray
            Array of shape (n_eras, n_features) with importance scores per era.

        Note:
        -----
        This is unique to WarpGBM's era-splitting capability. When no era_id
        is provided during training (standard ERM setting), n_eras=1 and this
        returns the same as get_feature_importance() but with shape (1, n_features).

        This allows you to see which features are invariant across eras vs
        which features are only important in specific eras.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")

        importance = self._per_era_importance(self._resolve_num_trees(num_trees)).copy()

        if normalize:
            for era_idx in range(importance.shape[0]):
                era_sum = importance[era_idx].sum()
                if era_sum > 0:
                    importance[era_idx] /= era_sum

        return importance

    def truncate(self, num_trees=None):
        """
        Drop all boosting rounds after ``num_trees`` (default: ``best_iteration_``).

        Use this before continuing training with ``warm_start=True`` from the best round of an
        early-stopped fit; without it, warm starts continue after all trained rounds.
        Training/eval histories are kept as logs. Per-era importances survive only when
        truncating to best_iteration_ (or with a single era).
        """
        self._check_fitted()
        if num_trees is None:
            if self.best_iteration_ is None:
                raise ValueError("truncate() without num_trees needs best_iteration_ (fit with an eval set).")
            num_trees = self.best_iteration_
        n = self._resolve_num_trees(num_trees)
        default_before = self._default_num_trees()

        era_importance = self._snapshot_importance(n)
        if era_importance is None and self._per_era_importance_all is not None \
                and self._per_era_importance_all.shape[0] == 1:
            era_importance = self._total_importance(n)[None, :]
        self._per_era_importance_all = era_importance

        if self._output_strategy() == PER_OUTPUT:
            del self.forest[n:]
        else:
            for r in range(n, len(self.forest)):
                self.forest[r] = {}
        self._trees_trained = n
        self._rollback = False
        if self.best_iteration_ is not None and self.best_iteration_ > n:
            self.best_iteration_ = None
            self.best_score_ = None
            self._per_era_importance_best = None
        if n != default_before:
            self.train_predictions_ = None

        self._invalidate_prediction_cache(keep_rounds=n)
        self.feature_importance_ = self._total_importance(n)
        self.per_era_feature_importance_ = era_importance
        return self

    # ─── Persistence ───

    def save_model(self, path):
        """
        Save the trained model to disk for later use.

        Parameters:
        -----------
        path : str
            File path where the model will be saved (e.g., 'model.pkl')

        Note:
        -----
        Custom objectives are pickled together with the model, so their class
        must be importable (defined at module level) when loading.

        Example:
        --------
        >>> model = WarpGBM(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>> model.save_model('checkpoint_100.pkl')
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")

        def to_cpu(value):
            return value.detach().cpu() if torch.is_tensor(value) else value

        # Collect all the necessary state to save
        state = {
            # Hyperparameters
            "objective": self.objective,
            "num_bins": self.num_bins,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "min_split_gain": self.min_split_gain,
            "min_sum_hessian": self.min_sum_hessian,
            "L2_reg": self.L2_reg,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "warm_start": self.warm_start,
            "monotonic_constraints": self.monotonic_constraints,
            "prebinned": self.prebinned,
            "record_training_loss": self.record_training_loss,
            "deterministic": self.deterministic,
            # Trained state
            "objective_": self.objective_,
            "forest": self.forest,
            "bin_edges": to_cpu(self.bin_edges),
            "prebinned_": self.prebinned_,
            "base_prediction": to_cpu(self.base_prediction),
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "num_outputs": self.num_outputs,
            "_targets_1d": self._targets_1d,
            "classes_": self.classes_,
            "label_encoder": self.label_encoder,
            "feature_importance_": self.feature_importance_,
            "per_era_feature_importance_": self.per_era_feature_importance_,
            "_is_fitted": self._is_fitted,
            "_trees_trained": self._trees_trained,
            "training_loss": (
                self.training_loss if hasattr(self, "training_loss") else []
            ),
            "eval_loss": self.eval_loss,
            "evals_result_": self.evals_result_,
            "eval_iterations_": self.eval_iterations_,
            "eval_metric_name_": self.eval_metric_name_,
            "eval_metric_higher_is_better_": self.eval_metric_higher_is_better_,
            "best_iteration_": self.best_iteration_,
            "best_score_": self.best_score_,
            "_rollback": self._rollback,
            "_per_era_importance_all": self._per_era_importance_all,
            "_per_era_importance_best": self._per_era_importance_best,
            "monotonic_constraints_vector_": self.monotonic_constraints_vector_,
            "monotonic_required_split_direction_": self.monotonic_required_split_direction_,
            "_monotonic_constraints_fitted_": self._monotonic_constraints_fitted_,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        if self.verbose:
            print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load a previously saved model from disk.

        Parameters:
        -----------
        path : str
            File path to the saved model (e.g., 'model.pkl')

        Returns:
        --------
        self : WarpGBM
            The model instance with loaded state

        Example:
        --------
        >>> model = WarpGBM()
        >>> model.load_model('checkpoint_100.pkl')
        >>> # Continue training with warm_start
        >>> model.warm_start = True
        >>> model.n_estimators = 200  # Train 100 more trees
        >>> model.fit(X_train, y_train)
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Restore hyperparameters
        self.objective = state['objective']
        self.num_bins = state['num_bins']
        self.max_depth = state['max_depth']
        self.learning_rate = state['learning_rate']
        self.n_estimators = state['n_estimators']
        self.min_child_weight = state['min_child_weight']
        self.min_split_gain = state['min_split_gain']
        self.min_sum_hessian = state.get('min_sum_hessian', 1e-3)
        self.L2_reg = state['L2_reg']
        self.colsample_bytree = state['colsample_bytree']
        self.random_state = state['random_state']
        self.warm_start = state['warm_start']
        self.monotonic_constraints = state.get("monotonic_constraints", None)
        self.prebinned = state.get("prebinned", None)
        # Older models used the training-time behaviour of their version: every round
        self.record_training_loss = state.get("record_training_loss", "all")
        self.deterministic = state.get("deterministic", True)

        # Restore trained state
        self.objective_ = state.get("objective_") or resolve_objective(self.objective)
        self.forest = state['forest']
        bin_edges = state['bin_edges']
        self.bin_edges = bin_edges.to(self.device) if torch.is_tensor(bin_edges) else bin_edges
        # Models saved before prebinned_ was stored keep binning with their (identity) edges
        self.prebinned_ = state.get("prebinned_", False)
        self.num_features = state['num_features']
        self.num_classes = state['num_classes']
        self.classes_ = state['classes_']
        self.label_encoder = state['label_encoder']
        if isinstance(self.objective_, Softmax) and self.num_classes is not None:
            self.objective_.num_classes = self.num_classes

        num_outputs = state.get("num_outputs")
        if num_outputs is None:
            # Models saved before num_outputs was stored
            num_outputs = self.num_classes if self.objective_.is_classifier else 1
        self.num_outputs = num_outputs
        self._targets_1d = state.get("_targets_1d", self.objective != "multi_regression")

        base = state['base_prediction']
        if torch.is_tensor(base):
            base = base.to(device=self.device, dtype=torch.float32)
            if base.numel() == 1 and not self.objective_.is_classifier:
                base = float(base.item())
        self.base_prediction = base

        self._is_fitted = state['_is_fitted']
        self._trees_trained = state['_trees_trained']
        self._objective_source = self.objective
        self.training_loss = state.get('training_loss', [])
        # Models saved before best-iteration tracking: all trees, no best round
        self.evals_result_ = state.get(
            "evals_result_", {"train": [], "eval": state.get("eval_loss", [])}
        )
        self.eval_loss = self.evals_result_["eval"]
        self.eval_iterations_ = state.get("eval_iterations_", [])
        self.eval_metric_name_ = state.get("eval_metric_name_")
        self.eval_metric_higher_is_better_ = state.get("eval_metric_higher_is_better_")
        self.best_iteration_ = state.get("best_iteration_")
        self.best_score_ = state.get("best_score_")
        self._rollback = state.get("_rollback", False)
        self._per_era_importance_all = state.get(
            "_per_era_importance_all", state['per_era_feature_importance_']
        )
        self._per_era_importance_best = state.get("_per_era_importance_best")
        self.feature_importance_ = state['feature_importance_']
        self.per_era_feature_importance_ = state['per_era_feature_importance_']
        self.train_predictions_ = None
        self.monotonic_constraints_vector_ = state.get(
            "monotonic_constraints_vector_", None
        )
        self.monotonic_required_split_direction_ = state.get(
            "monotonic_required_split_direction_", None
        )
        self._monotonic_constraints_fitted_ = state.get(
            "_monotonic_constraints_fitted_",
            self.monotonic_constraints_vector_ is not None,
        )

        self.monotonic_required_split_direction_gpu_ = None
        self._invalidate_prediction_cache()

        if self.verbose:
            print(f"Model loaded from {path} ({self._trees_trained} trees)")

        return self

    def flatten_tree(self, tree, max_nodes, output_index=None):
        flat = torch.full((max_nodes, _NODE_WIDTH), float("nan"), dtype=torch.float32)
        node_counter = [0]
        node_list = []

        def walk(node):
            curr_id = node_counter[0]
            node_counter[0] += 1

            new_node = {"node_id": curr_id}
            if "leaf_value" in node:
                leaf_value = node["leaf_value"]
                if isinstance(leaf_value, torch.Tensor):
                    leaf_value = leaf_value.detach().cpu().tolist()

                if isinstance(leaf_value, (list, tuple, np.ndarray)):
                    idx = 0 if output_index is None else output_index
                    new_node["leaf_value"] = float(leaf_value[idx])
                else:
                    new_node["leaf_value"] = float(leaf_value)
            else:
                new_node["best_feature"] = float(node["feature"])
                new_node["split_bin"] = float(node["bin"])
                new_node["nan_left"] = float(bool(node.get("nan_left", False)))
                new_node["left_id"] = node_counter[0]
                walk(node["left"])
                new_node["right_id"] = node_counter[0]
                walk(node["right"])

            node_list.append(new_node)
            return new_node

        walk(tree)

        for node in node_list:
            i = node["node_id"]
            if "leaf_value" in node:
                flat[i, 4] = 1.0
                flat[i, 5] = node["leaf_value"]
            else:
                flat[i, 0] = node["best_feature"]
                flat[i, 1] = node["split_bin"]
                flat[i, 2] = node["left_id"]
                flat[i, 3] = node["right_id"]
                flat[i, 4] = 0.0
                flat[i, 6] = node["nan_left"]

        return flat
