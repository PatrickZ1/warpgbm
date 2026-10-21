import inspect
import math
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
    _check_device,
    _prebinned_to_gpu,
    _quantile_edges,
    _to_float_gpu,
)
from warpgbm.cuda import _get_node_kernel
import gc


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


# Width of a flattened tree node, see warpgbm/cuda/predict.cu
_NODE_WIDTH = 7

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

    def __init__(
        self,
        objective="regression",
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20,
        min_split_gain=0.0,
        threads_per_block=64,
        rows_per_thread=4,
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
            threads_per_block=threads_per_block,
            rows_per_thread=rows_per_thread,
            L2_reg=L2_reg,
            device=device,
            colsample_bytree=colsample_bytree,
            monotonic_constraints=monotonic_constraints,
            min_sum_hessian=min_sum_hessian,
            prebinned=prebinned,
            record_training_loss=record_training_loss,
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
        self.threads_per_block = threads_per_block
        self.rows_per_thread = rows_per_thread
        self.L2_reg = L2_reg
        self.forest = [{} for _ in range(self.n_estimators)]
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.warm_start = warm_start
        self.monotonic_constraints = monotonic_constraints
        self.verbose = verbose
        self.prebinned = prebinned
        self.record_training_loss = record_training_loss
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
        self._cached_tree_tensors = None
        self._cached_tree_key = None
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
            "threads_per_block",
            "rows_per_thread",
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
        if kwargs["threads_per_block"] <= 0 or kwargs["threads_per_block"] % 32 != 0:
            raise ValueError(
                "threads_per_block should be a positive multiple of 32 (warp size)."
            )
        if not (1 <= kwargs["rows_per_thread"] <= 16):
            raise ValueError(
                "rows_per_thread must be positive between 1 and 16 inclusive."
            )
        if kwargs["L2_reg"] < 0:
            raise ValueError("L2_reg must be non-negative.")
        if kwargs["colsample_bytree"] <= 0 or kwargs["colsample_bytree"] > 1:
            raise ValueError(
                f"Invalid colsample_bytree: {kwargs['colsample_bytree']}. Must be a float value > 0 and <= 1."
            )

        if kwargs["prebinned"] not in (None, True, False):
            raise TypeError(f"prebinned must be None, True or False, got {kwargs['prebinned']!r}.")
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
        """Initial raw score as a [K] tensor on the model device."""
        K = self.num_outputs
        base = self.base_prediction
        if base is None:
            # Models saved before the base score of classifiers was stored started at 0
            return torch.zeros(K, dtype=torch.float32, device=self.device)
        base = torch.as_tensor(base, dtype=torch.float32).to(self.device).reshape(-1)
        if base.numel() == 1 and K > 1:
            base = base.expand(K).clone()
        return base

    def _store_base_score(self, base):
        """Keep the legacy representation: a float for single-output regression."""
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

    # ─── Prediction cache ───

    def _invalidate_prediction_cache(self):
        self._cached_tree_tensors = None
        self._cached_tree_key = None

    def _flatten_for_output(self, tree, max_nodes, output_index):
        # Single-output trees are flattened without output_index (keeps flatten_tree overridable)
        if output_index is None:
            return self.flatten_tree(tree, max_nodes=max_nodes)
        return self.flatten_tree(tree, max_nodes=max_nodes, output_index=output_index)

    def _trees_per_output(self, rounds):
        """Group fitted trees by score column: returns K lists of (tree, output_index)."""
        K = self.num_outputs
        groups = [[] for _ in range(K)]
        if self._output_strategy() == PER_OUTPUT:
            for round_trees in rounds:
                if not round_trees:
                    continue
                for k, tree in enumerate(round_trees):
                    if tree:
                        groups[k].append((tree, None))
        else:
            for tree in rounds:
                if not tree:
                    continue
                for k in range(K):
                    groups[k].append((tree, None if K == 1 else k))
        return groups

    def _flatten_groups(self, groups):
        max_nodes = 2 ** (self.max_depth + 1)
        tensors = []
        for group in groups:
            if group:
                tensors.append(
                    torch.stack(
                        [
                            self._flatten_for_output(tree, max_nodes, out_idx)
                            for tree, out_idx in group
                        ]
                    ).to(self.device)
                )
            else:
                tensors.append(None)
        return tensors

    def _get_cached_tree_tensors(self, num_trees):
        max_nodes = 2 ** (self.max_depth + 1)
        key = (
            str(self.device),
            max_nodes,
            self._output_strategy(),
            num_trees,
            self.num_outputs,
        )
        if self._cached_tree_tensors is not None and self._cached_tree_key == key:
            return self._cached_tree_tensors

        rounds = self.forest[:num_trees]
        tensors = self._flatten_groups(self._trees_per_output(rounds))
        self._cached_tree_tensors = tensors
        self._cached_tree_key = key
        return tensors

    def _accumulate_tree_tensors(self, bin_indices, tree_tensors, out):
        """Add learning_rate * sum of trees to out[:, k] for every score column k."""
        bin_indices = bin_indices.contiguous()
        for k, tree_tensor in enumerate(tree_tensors):
            if tree_tensor is None or tree_tensor.size(0) == 0:
                continue
            col = torch.zeros(out.shape[0], device=self.device, dtype=torch.float32)
            _get_node_kernel().predict_forest(
                bin_indices,
                tree_tensor.contiguous(),
                self.nan_bin,
                self.learning_rate,
                col,
            )
            out[:, k] += col
        return out

    # ─── Fit ───

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

        self.Y_gpu, self.tw = self._build_targets(y_prepared, sample_weight)
        self.row_weight_gpu = self.tw.row_weight.contiguous()
        self._count_scale = self._fixed_point_scale(
            self.row_weight_gpu, self.threads_per_block * self.rows_per_thread
        )

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

        self.scores = self._predict_raw_binned(self.bin_indices, self._trees_trained)
        self._best_train_scores = None
        self._train_aux = self._aux_tensor(train_aux)

        self.root_node_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
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
            self._eval_scores = self._predict_raw_binned(self.bin_indices_eval, self._trees_trained)
        else:
            self.bin_indices_eval = None
            self.Y_gpu_eval = None
            self.tw_eval = None
            self.eval_every_n_trees = None
            self.early_stopping_rounds = None
            self._eval_aux = None
            self._eval_scores = None

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

        gc.collect()

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

    def compute_histograms(self, sample_indices, feature_indices, grad, hess):
        num_hist_bins = self.num_bins + 1  # value bins + NaN bin
        num_channels = grad.shape[1]
        # The kernel overwrites its outputs
        grad_hist = torch.empty(
            (self.num_eras, len(feature_indices), num_hist_bins, num_channels),
            device=self.device,
            dtype=torch.float32,
        )
        hess_hist = torch.empty_like(grad_hist)
        count_hist = torch.empty(
            (self.num_eras, len(feature_indices), num_hist_bins),
            device=self.device,
            dtype=torch.float32,
        )

        _get_node_kernel().compute_histogram3(
            self.bin_indices,
            grad,
            hess,
            self.row_weight_gpu,
            sample_indices,
            feature_indices,
            self.era_indices,
            grad_hist,
            hess_hist,
            count_hist,
            num_hist_bins,
            *self._hist_scales,
            self.threads_per_block,
            self.rows_per_thread,
        )
        return grad_hist, hess_hist, count_hist

    @staticmethod
    def _fixed_point_scale(values, rows_per_block):
        """
        Power-of-two scale for the deterministic fixed-point histogram accumulation
        (see warpgbm/cuda/histogram_kernel.cu): the largest scale for which the sum of
        one CUDA block's rows cannot overflow the 32-bit shared-memory accumulators.
        """
        bound = float(values.abs().max()) * rows_per_block if values.numel() else 0.0
        if not math.isfinite(bound) or bound <= 0.0:
            return 1.0
        exponent = math.floor(math.log2(2.0**30 / bound))
        return 2.0 ** max(-126, min(126, exponent))

    def find_best_split(self, gradient_histogram, hessian_histogram, count_histogram):
        _get_node_kernel().compute_split(
            gradient_histogram,
            hessian_histogram,
            count_histogram,
            float(self.min_child_weight),
            float(self.min_sum_hessian),
            self.L2_reg,
            self.per_era_gain,
            self.per_era_direction,
            self.threads_per_block,
        )

        if self.num_eras == 1:
            era_splitting_criterion = self.per_era_gain[0,:,:]  # [F, 2 * num_bins]
            dir_score_mask = era_splitting_criterion > self.min_split_gain
        else:
            directional_agreement = self.per_era_direction.mean(dim=0).abs()  # [F, 2 * num_bins]
            era_splitting_criterion = self.per_era_gain.mean(dim=0)  # [F, 2 * num_bins]
            dir_score_mask = ( directional_agreement == directional_agreement.max() ) & (era_splitting_criterion > self.min_split_gain)

        # Checked on the host once per tree, so unconstrained models skip this without a sync
        if self._tree_has_monotone:
            local_required_dirs = self.monotonic_required_split_direction_gpu_[
                self.feat_indices_tree.to(torch.long)
            ]
            row_mask = (local_required_dirs != 0).view(-1, 1)
            required = local_required_dirs.to(self.per_era_direction.dtype).view(-1, 1)

            if self.num_eras == 1:
                local_dirs = self.per_era_direction[0, :, :]
                monotonic_mask = (~row_mask) | (local_dirs == required)
            else:
                all_era_matches = (
                    self.per_era_direction == required.unsqueeze(0)
                ).all(dim=0)
                monotonic_mask = (~row_mask) | all_era_matches

            dir_score_mask = dir_score_mask & monotonic_mask

        masked_criterion = torch.where(
            dir_score_mask, era_splitting_criterion, float("-inf")
        )
        # max() returns the first maximal index, like argmax; fetch value and index in one sync
        best_val, best_idx = torch.max(masked_criterion.view(-1), dim=0)
        best_val, best_idx = torch.stack([best_val.double(), best_idx.double()]).tolist()
        if best_val == float("-inf"):
            return -1, -1

        best_feature, best_candidate = divmod(int(best_idx), 2 * self.num_bins)
        return best_feature, best_candidate

    def _go_left(self, bins, split_bin, nan_left):
        """Routing rule shared by training and prediction (see predict.cu)."""
        # The NaN bin lies above every value bin, so `bins <= split_bin` already sends NaN right
        go_left = bins <= split_bin
        if nan_left:
            go_left |= bins == self.nan_bin
        return go_left

    def _leaf_value(self, node_indices):
        gh = torch.sum(self._tree_grad_hess[node_indices], dim=0)
        num_channels = gh.shape[0] // 2
        return gh[:num_channels] / (gh[num_channels:] + self.L2_reg)

    def _make_leaf(self, node_indices, lower, upper):
        leaf_value = self._leaf_value(node_indices)
        if self._tree_has_monotone:
            leaf_value = torch.clamp(leaf_value, min=lower, max=upper)

        if self._tree_score_col is not None:
            # One tree per score column: update that column only
            self.scores[node_indices, self._tree_score_col] += (
                self.learning_rate * leaf_value[0]
            )
        else:
            self.scores[node_indices] += self.learning_rate * leaf_value

        # Leaf values are copied to the host once per tree (see _grow_single_tree)
        leaf = {"leaf_value": None, "samples": node_indices.numel()}
        self._pending_leaves.append((leaf, leaf_value))
        return leaf

    def grow_tree(
        self,
        gradient_histogram,
        hessian_histogram,
        count_histogram,
        node_indices,
        depth,
        lower,
        upper,
    ):
        if depth == self.max_depth:
            return self._make_leaf(node_indices, lower, upper)

        local_feature, best_candidate = self.find_best_split(
            gradient_histogram, hessian_histogram, count_histogram
        )

        if local_feature == -1:
            return self._make_leaf(node_indices, lower, upper)

        nan_left = best_candidate >= self.num_bins
        best_bin = best_candidate % self.num_bins

        # Track feature importance: accumulate per-era gains for the chosen feature
        global_feature_idx = self._feat_indices_cpu[local_feature]
        split_gain = self.per_era_gain[:, local_feature, best_candidate]  # [num_eras]
        self._feature_importance_gpu[:, global_feature_idx] += split_gain
        split_gain_total = split_gain.sum()

        split_mask = self._go_left(
            self.bin_indices[node_indices, global_feature_idx], best_bin, nan_left
        )
        left_indices = node_indices[split_mask]
        right_indices = node_indices[~split_mask]

        # ─── Monotonic constraints: propagate value bounds to the children ───
        left_lower, left_upper = lower, upper
        right_lower, right_upper = lower, upper
        constraint = int(self.monotonic_constraints_vector_[global_feature_idx])
        if constraint != 0:
            left_val = torch.clamp(self._leaf_value(left_indices), min=lower, max=upper)
            right_val = torch.clamp(self._leaf_value(right_indices), min=lower, max=upper)
            mid = 0.5 * (left_val + right_val)
            if constraint > 0:
                left_upper = torch.minimum(upper, mid)
                right_lower = torch.maximum(lower, mid)
            else:
                left_lower = torch.maximum(lower, mid)
                right_upper = torch.minimum(upper, mid)

        if left_indices.numel() <= right_indices.numel():
            grad_hist_left, hess_hist_left, count_hist_left = self.compute_histograms(
                left_indices,
                self.feat_indices_tree,
                self._tree_grad,
                self._tree_hess,
            )
            grad_hist_right = gradient_histogram - grad_hist_left
            hess_hist_right = hessian_histogram - hess_hist_left
            count_hist_right = count_histogram - count_hist_left
        else:
            grad_hist_right, hess_hist_right, count_hist_right = self.compute_histograms(
                right_indices,
                self.feat_indices_tree,
                self._tree_grad,
                self._tree_hess,
            )
            grad_hist_left = gradient_histogram - grad_hist_right
            hess_hist_left = hessian_histogram - hess_hist_right
            count_hist_left = count_histogram - count_hist_right

        new_depth = depth + 1
        left_child = self.grow_tree(
            grad_hist_left,
            hess_hist_left,
            count_hist_left,
            left_indices,
            new_depth,
            left_lower,
            left_upper,
        )
        right_child = self.grow_tree(
            grad_hist_right,
            hess_hist_right,
            count_hist_right,
            right_indices,
            new_depth,
            right_lower,
            right_upper,
        )

        node = {
            "feature": global_feature_idx,
            "bin": best_bin,
            "nan_left": bool(nan_left),
            "left": left_child,
            "right": right_child,
        }
        self._pending_gains.append((node, split_gain_total))
        return node

    def _grow_single_tree(self, grad, hess, score_col=None):
        """Grow one tree on gradient channels grad/hess [N, Kc] (negative-gradient convention)."""
        self._tree_grad = grad
        self._tree_hess = hess
        self._tree_grad_hess = torch.cat([grad, hess], dim=1)  # one gather per leaf
        self._tree_score_col = score_col

        self._feat_indices_cpu = self.feat_indices_tree.tolist()
        self._tree_has_monotone = bool(
            np.any(self.monotonic_constraints_vector_[self._feat_indices_cpu])
        )
        self._pending_leaves = []
        self._pending_gains = []

        rows_per_block = self.threads_per_block * self.rows_per_thread
        self._hist_scales = (
            self._fixed_point_scale(grad, rows_per_block),
            self._fixed_point_scale(hess, rows_per_block),
            self._count_scale,
        )

        num_channels = grad.shape[1]
        lower = torch.full((num_channels,), float("-inf"), device=self.device)
        upper = torch.full((num_channels,), float("inf"), device=self.device)

        root_grad_hist, root_hess_hist, root_count_hist = self.compute_histograms(
            self.root_node_indices,
            self.feat_indices_tree,
            grad,
            hess,
        )
        tree = self.grow_tree(
            root_grad_hist,
            root_hess_hist,
            root_count_hist,
            self.root_node_indices,
            0,
            lower,
            upper,
        )
        self._tree_grad = None
        self._tree_hess = None
        self._tree_grad_hess = None

        # One device -> host transfer per tree: leaf values and split gains
        leaf_values = torch.stack([v for _, v in self._pending_leaves])  # [L, Kc]
        num_leaves, num_channels = leaf_values.shape
        packed = [leaf_values.reshape(-1)]
        if self._pending_gains:
            packed.append(torch.stack([g for _, g in self._pending_gains]).to(leaf_values.dtype))
        host = torch.cat(packed).cpu().tolist()
        for j, (leaf, _) in enumerate(self._pending_leaves):
            value = host[j * num_channels : (j + 1) * num_channels]
            leaf["leaf_value"] = value[0] if num_channels == 1 else value
        for (node, _), gain in zip(self._pending_gains, host[num_leaves * num_channels :]):
            node["gain"] = gain
        self._pending_leaves = []
        self._pending_gains = []
        return tree

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
        self._best_importance_gpu = None
        strategy = self._output_strategy()

        if self.colsample_bytree < 1.0:
            k = max(1, int(self.colsample_bytree * self.num_features))
        else:
            self.feat_indices_tree = self.feature_indices
            k = self.num_features

        self._feature_importance_gpu = torch.zeros(
            (self.num_eras, self.num_features), device=self.device, dtype=torch.float32
        )

        num_candidates = 2 * self.num_bins
        self.per_era_gain = torch.zeros(self.num_eras, k, num_candidates, device=self.device, dtype=torch.float32)
        self.per_era_direction = torch.zeros(self.num_eras, k, num_candidates, device=self.device, dtype=torch.float32)

        # Warm start: start from where we left off
        start_iter = self._trees_trained if self.warm_start and self._is_fitted else 0

        if strategy == PER_OUTPUT:
            del self.forest[start_iter:]
        elif len(self.forest) < self.n_estimators:
            # Ensure forest has enough slots
            self.forest.extend([{} for _ in range(self.n_estimators - len(self.forest))])

        for i in range(start_iter, self.n_estimators):
            g, h = compute_grad_hess(
                self.objective_, self.Y_gpu, self.scores, self.tw.weight, self.tw.mask
            )
            # Trees are grown on the negative gradient: leaf = sum(-g) / (sum(h) + L2)
            neg_grad = -g

            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(
                    self.num_features, device=self.device, dtype=torch.int32, generator=self._rng
                )[:k]

            if strategy == PER_OUTPUT:
                round_trees = []
                for col in range(self.num_outputs):
                    round_trees.append(
                        self._grow_single_tree(
                            neg_grad[:, col : col + 1].contiguous(),
                            h[:, col : col + 1].contiguous(),
                            score_col=col,
                        )
                    )
                self.forest.append(round_trees)
            else:
                round_trees = self._grow_single_tree(neg_grad, h)
                self.forest[i] = round_trees

            self._trees_trained = i + 1
            self._invalidate_prediction_cache()

            self.compute_eval(i, round_trees)

            if self.stop:
                break

        # Per-era importance of all trees, and of the best round of this fit
        if base_known:
            self._per_era_importance_all = base + self._feature_importance_gpu.cpu().numpy()
            if self._best_importance_gpu is not None:
                self._per_era_importance_best = base + self._best_importance_gpu.cpu().numpy()
        self._feature_importance_gpu = None
        self._best_importance_gpu = None
        self._is_fitted = True
        self._update_importance_attributes()

        if self.verbose:
            if strategy == PER_OUTPUT:
                print(f"Finished training {self.objective_.name} forest. Total rounds: {self._trees_trained} ({self._trees_trained * self.num_outputs} trees)")
            else:
                print(f"Finished training {self.objective_.name} forest. Total trees: {self._trees_trained}")

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

    def _add_round_to_eval_scores(self, round_trees):
        """Incrementally add the newest round to the eval scores (avoids re-flattening the forest)."""
        groups = self._trees_per_output([round_trees])
        self._accumulate_tree_tensors(
            self.bin_indices_eval, self._flatten_groups(groups), self._eval_scores
        )

    def compute_eval(self, i, round_trees):
        if self.eval_every_n_trees is None:
            return

        self._add_round_to_eval_scores(round_trees)

        obj = self.objective_
        loss_name = "loss" if obj.has_loss() else "mse"
        # The training loss costs a full pass over the training set and a host sync, so by
        # default it is only computed at evaluation checkpoints
        train_loss = None
        if self.record_training_loss == "all":
            train_loss = self._metric_value(loss_name, self.Y_gpu, self.scores, self.tw)
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
            train_score = self._metric_value(metric, self.Y_gpu, self.scores, self.tw, self._train_aux)
            self.evals_result_["train"].append(train_score)
        self.evals_result_["eval"].append(eval_score)
        self.eval_iterations_.append(i + 1)

        if self.record_training_loss == "eval":
            if isinstance(metric, str) and metric == loss_name and train_score is not None:
                train_loss = train_score
            else:
                train_loss = self._metric_value(loss_name, self.Y_gpu, self.scores, self.tw)
            self.training_loss.append(train_loss)

        if self._is_better(eval_score, self.best_score_):
            self.best_score_ = eval_score
            self.best_iteration_ = i + 1
            self._evals_since_best = 0
            self._best_importance_gpu = self._feature_importance_gpu.clone()
            if self._rollback:
                self._best_train_scores = self.scores.clone()
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

    def bin_data_with_existing_edges(self, X_np):
        """Internal int8 bins [N, F] of raw features X_np with the fitted bin edges."""
        if self.bin_edges is None:
            raise ValueError("This model has no bin edges (it was fitted on pre-binned data).")
        with torch.no_grad():
            X_gpu = _to_float_gpu(X_np, self.device)
            return _bin_matrix(X_gpu, self.bin_edges.to(self.device), self.nan_bin)

    def bin_inference_data(self, X_np, name="X"):
        """
        Internal int8 bins [N, F] on the model device: raw features are binned with the fitted
        edges; for a pre-binned model (prebinned_), X_np must hold bin indices (-1 = missing).
        """
        if X_np.ndim != 2 or X_np.shape[1] != self.num_features:
            raise ValueError(
                f"{name} must have shape (n_samples, {self.num_features}), got {X_np.shape}."
            )
        if self.prebinned_:
            return _prebinned_to_gpu(X_np, self.num_bins, self.device, name)
        return self.bin_data_with_existing_edges(X_np)

    def _predict_raw_binned(self, bin_indices, num_trees=None):
        """Raw scores [N, K] for binned data, using the first num_trees rounds (None: default model)."""
        num_trees = self._resolve_num_trees(num_trees)
        num_samples = bin_indices.size(0)
        out = (
            torch.zeros(
                (num_samples, self.num_outputs), device=self.device, dtype=torch.float32
            )
            + self._base_score_tensor()
        )
        if num_trees > 0:
            self._accumulate_tree_tensors(
                bin_indices, self._get_cached_tree_tensors(num_trees), out
            )
        return out

    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

    def predict_binned(self, bin_indices, num_trees=None):
        """Transformed predictions for binned data ([N] for single-output targets)."""
        pred = self.objective_.transform(self._predict_raw_binned(bin_indices, num_trees))
        if not self.objective_.is_classifier and self._targets_1d and pred.shape[1] == 1:
            return pred[:, 0]
        return pred

    def predict_raw(self, X_np, num_trees=None):
        """
        Raw (untransformed) scores, e.g. logits for classification.

        num_trees: number of boosting rounds to use. None uses the default model, i.e.
        best_iteration_ after early stopping and all trained rounds otherwise.

        Returns shape (n_samples,) for single-output regression-like models, else (n_samples, K).
        """
        self._check_fitted()
        num_trees = self._resolve_num_trees(num_trees)
        bin_indices = self.bin_inference_data(X_np)
        raw = self._predict_raw_binned(bin_indices, num_trees).cpu().numpy()
        del bin_indices
        if not self.objective_.is_classifier and self._targets_1d and raw.shape[1] == 1:
            return raw.reshape(-1)
        return raw

    def predict(self, X_np, num_trees=None):
        """
        Predict on new data.

        For regression / custom objectives: returns objective.transform(raw scores)
        For classification: returns predicted class labels

        num_trees: number of boosting rounds to use. None uses the default model, i.e.
        best_iteration_ after early stopping and all trained rounds otherwise.
        """
        self._check_fitted()
        num_trees = self._resolve_num_trees(num_trees)
        if self.objective_.is_classifier:
            # Classification: return class labels
            probs = self.predict_proba(X_np, num_trees)
            class_indices = np.argmax(probs, axis=1)
            return self.label_encoder.inverse_transform(class_indices)

        bin_indices = self.bin_inference_data(X_np)
        preds = self.predict_binned(bin_indices, num_trees).cpu().numpy()
        del bin_indices
        return preds

    def predict_proba(self, X_np, num_trees=None):
        """
        Predict class probabilities (classification only).

        num_trees: number of boosting rounds to use (None: default model, see predict).

        Returns:
            np.array of shape (n_samples, n_classes) with probabilities
        """
        if not self._is_classifier():
            raise ValueError("predict_proba only available for classification objectives")
        self._check_fitted()
        num_trees = self._resolve_num_trees(num_trees)

        bin_indices = self.bin_inference_data(X_np)
        probs = self.predict_proba_binned(bin_indices, num_trees).cpu().numpy()
        del bin_indices
        return probs

    def predict_proba_binned(self, bin_indices, num_trees=None):
        """
        Predict probabilities from pre-binned data (classification).

        Returns:
            torch.Tensor of shape (n_samples, n_classes) with probabilities
        """
        return self.objective_.transform(self._predict_raw_binned(bin_indices, num_trees))

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

        self._invalidate_prediction_cache()
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
