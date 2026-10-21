import inspect
import math
import torch
import numpy as np
import pickle
import importlib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from warpgbm.metrics import (
    weighted_multi_mse_torch,
    weighted_rmsle_torch,
    weighted_log_loss_torch,
    weighted_accuracy_torch,
    weighted_corr_loss_torch,
)
from warpgbm.objectives import (
    Objective,
    ObjectiveContext,
    Softmax,
    PER_OUTPUT,
    VECTOR,
    compute_grad_hess,
    resolve_objective,
    weighted_loss,
)
import gc

_NODE_KERNEL = None
_NODE_KERNEL_IMPORT_ERROR = None

# Width of a flattened tree node, see warpgbm/cuda/predict.cu
_NODE_WIDTH = 7

_REGRESSION_METRICS = ["mse", "corr", "rmsle", "loss"]
_MULTI_OUTPUT_METRICS = ["mse", "loss"]
_CLASSIFICATION_METRICS = ["logloss", "accuracy", "loss"]


def _get_node_kernel():
    global _NODE_KERNEL
    global _NODE_KERNEL_IMPORT_ERROR

    if _NODE_KERNEL is not None:
        return _NODE_KERNEL

    if _NODE_KERNEL_IMPORT_ERROR is not None:
        raise ImportError(
            "WarpGBM CUDA extension 'warpgbm.cuda.node_kernel' is not available. "
            "Build/install with CUDA-enabled PyTorch and CUDA toolkit, then run "
            "`python setup.py build_ext --inplace`."
        ) from _NODE_KERNEL_IMPORT_ERROR

    try:
        _NODE_KERNEL = importlib.import_module("warpgbm.cuda.node_kernel")
    except ImportError as e:
        _NODE_KERNEL_IMPORT_ERROR = e
        raise ImportError(
            "WarpGBM CUDA extension 'warpgbm.cuda.node_kernel' is not available. "
            "Build/install with CUDA-enabled PyTorch and CUDA toolkit, then run "
            "`python setup.py build_ext --inplace`."
        ) from e

    return _NODE_KERNEL


class WarpGBM(BaseEstimator, RegressorMixin):

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
        era_splitting=True,
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
            colsample_bytree=colsample_bytree,
            monotonic_constraints=monotonic_constraints,
            min_sum_hessian=min_sum_hessian,
            era_splitting=era_splitting,
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
        self.era_splitting = era_splitting
        self.threads_per_block = threads_per_block
        self.rows_per_thread = rows_per_thread
        self.L2_reg = L2_reg
        self.forest = [{} for _ in range(self.n_estimators)]
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.warm_start = warm_start
        self.monotonic_constraints = monotonic_constraints
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

    def _validate_hyperparams(self, **kwargs):
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

        if not (2 <= kwargs["num_bins"] <= 127):
            raise ValueError("num_bins must be between 2 and 127 inclusive.")
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

        if not isinstance(kwargs.get("era_splitting", True), bool):
            raise TypeError("era_splitting must be a bool.")

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
    def nan_bin(self):
        """Bin index reserved for missing (NaN) feature values."""
        return self.num_bins

    def _resolve_objective(self):
        if not (self.warm_start and self._is_fitted and self.objective_ is not None):
            return resolve_objective(self.objective)

        old = self.objective_
        unchanged = (
            self.objective is old
            or (isinstance(self.objective, str) and self.objective == old.name)
        )
        if unchanged:
            return old

        # Warm start with a different objective (e.g. MSE warm-up, then a custom loss):
        # the raw score space and the forest layout must stay compatible.
        new = resolve_objective(self.objective)
        if new.output_strategy != old.output_strategy or new.is_classifier != old.is_classifier:
            raise ValueError(
                f"Cannot warm-start {new!r} from a model trained with {old!r}: "
                "output_strategy and is_classifier must match."
            )
        if isinstance(new, Softmax) and isinstance(old, Softmax):
            new.num_classes = old.num_classes
        return new

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
        era_id_eval=None,
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
        if era_id_eval is not None:
            if X_eval is None or y_eval is None:
                raise ValueError(
                    "era_id_eval can only be used when X_eval and y_eval are provided."
                )
            if not isinstance(era_id_eval, np.ndarray) or era_id_eval.ndim != 1:
                raise TypeError("era_id_eval must be a 1-dimensional numpy array.")
            if len(era_id_eval) != X_eval.shape[0]:
                raise ValueError(
                    f"era_id_eval must have same length as y_eval. Got {len(era_id_eval)} and {X_eval.shape[0]}."
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

        return early_stopping_rounds  # May have been defaulted here

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

    def _build_context(self, y_prepared, sample_weight, era_index):
        """
        Build GPU targets and the ObjectiveContext.

        Args:
            y_prepared: [N, T] float32 numpy targets, NaN/inf = missing.
            sample_weight: None, [N] or [N, T] numpy weights (already normalized).
            era_index: [N] int32 tensor.

        Returns:
            (Y_gpu [N, T] with missing entries filled with 0, ObjectiveContext)
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

        ctx = ObjectiveContext(
            weight=weight.to(torch.float32).contiguous(),
            mask=mask,
            target_mask=target_mask,
            era_index=era_index,
            num_outputs=K,
        )
        return Y_gpu, ctx

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

    def _get_cached_tree_tensors(self):
        max_nodes = 2 ** (self.max_depth + 1)
        key = (
            str(self.device),
            max_nodes,
            self._output_strategy(),
            self._trees_trained,
            self.num_outputs,
        )
        if self._cached_tree_tensors is not None and self._cached_tree_key == key:
            return self._cached_tree_tensors

        rounds = self.forest[: self._trees_trained]
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
        era_id_eval=None,
    ):
        self.objective_ = self._resolve_objective()

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
            era_id_eval,
        )

        if sample_weight is not None:
            sample_weight = self._normalize_sample_weight(sample_weight)
        if X_eval is not None and y_eval is not None and sample_weight_eval is not None:
            sample_weight_eval = self._normalize_sample_weight(sample_weight_eval)

        # Set random seed for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
                torch.cuda.manual_seed_all(self.random_state)
                # Note: Some CUDA operations may still be non-deterministic
                # For full determinism, set: torch.use_deterministic_algorithms(True)
                # but this may impact performance

        continuing = self.warm_start and self._is_fitted

        # ─── Warm Start Logic ───
        if not continuing:
            # Fresh start: reset training state
            self._is_fitted = False
            self._trees_trained = 0
            self.training_loss = []
            self.eval_loss = []
            self._invalidate_prediction_cache()
        else:
            # Continuing training: validate that data is compatible
            if X.shape[1] != self.num_features:
                raise ValueError(
                    f"X has {X.shape[1]} features, but model was trained with {self.num_features} features."
                )
            if self._trees_trained >= self.n_estimators:
                # Already have all requested trees
                print(f"Model already has {self._trees_trained} trees (n_estimators={self.n_estimators}). No additional training needed.")
                return self

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
        # The objective always sees the eras (ctx.era_index, e.g. for per-era losses);
        # directional era splitting in the trees can be switched off independently.
        loss_era_index = self.era_indices.to(dtype=torch.int32)
        if self.era_splitting:
            self.num_eras = len(self.unique_eras)
            self.era_indices = loss_era_index
        else:
            self.num_eras = 1
            self.era_indices = torch.zeros_like(loss_era_index)

        self.Y_gpu, self.ctx = self._build_context(
            y_prepared, sample_weight, loss_era_index
        )
        self.row_weight_gpu = self.ctx.row_weight.contiguous()
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
                self.objective_.init_score(self.Y_gpu, self.ctx),
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

        self.scores = self._predict_raw_binned(self.bin_indices)

        self.root_node_indices = torch.arange(self.num_samples, device=self.device, dtype=torch.int32)
        self.feature_indices = torch.arange(self.num_features, device=self.device, dtype=torch.int32)

        # ─── Optional Eval Set ───
        if X_eval is not None and y_eval is not None:
            self.bin_indices_eval = self.bin_inference_data(X_eval)
            if era_id_eval is None:
                era_index_eval = torch.zeros(X_eval.shape[0], dtype=torch.int32, device=self.device)
            else:
                _, era_index_eval = torch.unique(
                    torch.from_numpy(era_id_eval).to(self.device), return_inverse=True
                )
                era_index_eval = era_index_eval.to(torch.int32)
            self.Y_gpu_eval, self.ctx_eval = self._build_context(
                y_eval_prepared, sample_weight_eval, era_index_eval
            )
            self.eval_every_n_trees = eval_every_n_trees
            self.early_stopping_rounds = early_stopping_rounds
            self.eval_metric = self._resolve_eval_metric(eval_metric)
            self._eval_scores = self._predict_raw_binned(self.bin_indices_eval)
        else:
            self.bin_indices_eval = None
            self.Y_gpu_eval = None
            self.ctx_eval = None
            self.eval_every_n_trees = None
            self.early_stopping_rounds = None
            self._eval_scores = None

        # ─── Grow the forest ───
        with torch.no_grad():
            self.grow_forest()

        # Release training buffers
        self.bin_indices = None
        self.Y_gpu = None
        self.ctx = None
        self.scores = None
        self.row_weight_gpu = None
        self.bin_indices_eval = None
        self.Y_gpu_eval = None
        self.ctx_eval = None
        self._eval_scores = None

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

    def preprocess_gpu_data(self, X_np, era_id_np, reuse_edges=False):
        with torch.no_grad():
            self.num_samples, self.num_features = X_np.shape

            era_id_gpu = torch.from_numpy(era_id_np).to(
                dtype=torch.int32, device=self.device
            )
            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)

            if reuse_edges and self.bin_edges is not None:
                self.bin_edges = self.bin_edges.to(self.device)
                bin_indices = self.bin_inference_data(X_np)
                return bin_indices, era_indices, self.bin_edges, unique_eras

            is_integer_type = np.issubdtype(X_np.dtype, np.integer)

            if is_integer_type and np.all(X_np.max(axis=0) < self.num_bins):
                print("Detected pre-binned integer input — skipping quantile binning.")
                bin_indices = torch.as_tensor(X_np, device=self.device).to(
                    dtype=torch.int8
                ).contiguous()

                # Identity edges so that integer bins map onto themselves at predict-time
                bin_edges = torch.arange(
                    1, self.num_bins, dtype=torch.float32
                ).repeat(self.num_features, 1)
                bin_edges = bin_edges.to(self.device)
                return bin_indices, era_indices, bin_edges, unique_eras

            bin_edges = torch.empty(
                (self.num_features, self.num_bins - 1),
                dtype=torch.float32,
                device=self.device,
            )

            X_float_t = (
                torch.as_tensor(X_np, device=self.device, dtype=torch.float32)
                .transpose(0, 1)
                .contiguous()
            )
            bin_indices_t = torch.empty(
                (self.num_features, self.num_samples),
                dtype=torch.int8,
                device=self.device,
            )
            quantiles = torch.linspace(
                0, 1, self.num_bins + 1, device=self.device, dtype=torch.float32
            )[1:-1]

            for f in range(self.num_features):
                X_f = X_float_t[f]
                # NaNs are excluded from the edges and get their own bin
                bin_edges_f = torch.nanquantile(X_f, quantiles, dim=0)
                bin_edges_f = torch.nan_to_num(bin_edges_f, nan=0.0).contiguous()
                _get_node_kernel().custom_cuda_binner(
                    X_f, bin_edges_f, bin_indices_t[f], self.nan_bin
                )
                bin_edges[f, :] = bin_edges_f

            bin_indices = bin_indices_t.transpose(0, 1).contiguous()
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
        self._feature_importance_gpu[:, global_feature_idx] += self.per_era_gain[
            :, local_feature, best_candidate
        ]  # [num_eras]

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

        return {
            "feature": global_feature_idx,
            "bin": best_bin,
            "nan_left": bool(nan_left),
            "left": left_child,
            "right": right_child,
        }

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

        values = torch.stack([v for _, v in self._pending_leaves]).cpu().tolist()
        for (leaf, _), value in zip(self._pending_leaves, values):
            leaf["leaf_value"] = value[0] if len(value) == 1 else value
        self._pending_leaves = []
        return tree

    def grow_forest(self):
        """Gradient boosting loop shared by all objectives."""
        # Warm start: preserve existing training state
        if not hasattr(self, 'training_loss') or not self.warm_start or not self._is_fitted:
            self.training_loss = []
            self.eval_loss = []  # if eval set is given
            self.per_era_feature_importance_ = np.zeros((self.num_eras, self.num_features), dtype=np.float32)
        elif self.per_era_feature_importance_.shape != (self.num_eras, self.num_features):
            # Warm start with a different era layout: keep the totals in a single row
            totals = self.per_era_feature_importance_.sum(axis=0, keepdims=True)
            self.per_era_feature_importance_ = np.zeros((self.num_eras, self.num_features), dtype=np.float32)
            self.per_era_feature_importance_[0] = totals[0]

        self.stop = False
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
            self.ctx.iteration = i
            g, h = compute_grad_hess(self.objective_, self.Y_gpu, self.scores, self.ctx)
            # Trees are grown on the negative gradient: leaf = sum(-g) / (sum(h) + L2)
            neg_grad = -g

            if self.colsample_bytree < 1.0:
                self.feat_indices_tree = torch.randperm(self.num_features, device=self.device, dtype=torch.int32)[:k]

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

        # Aggregate feature importance across eras
        self.per_era_feature_importance_ += self._feature_importance_gpu.cpu().numpy()
        self._feature_importance_gpu = None
        self.feature_importance_ = self.per_era_feature_importance_.sum(axis=0)
        self._is_fitted = True

        if strategy == PER_OUTPUT:
            print(f"Finished training {self.objective_.name} forest. Total rounds: {self._trees_trained} ({self._trees_trained * self.num_outputs} trees)")
        else:
            print(f"Finished training {self.objective_.name} forest. Total trees: {self._trees_trained}")

    # ─── Evaluation ───

    def _metric_value(self, metric, Y, raw, ctx):
        """Evaluate a metric (lower is better) on raw scores raw [N, K]."""
        obj = self.objective_
        if metric == "loss":
            return weighted_loss(obj, Y, raw, ctx)

        pred = obj.transform(raw)
        target_weight = ctx.weight if ctx.weight.shape == ctx.target_mask.shape else ctx.row_weight[:, None]

        if callable(metric):
            y_true, mask, weight = Y, ctx.target_mask, target_weight
            if obj.is_classifier:
                y_true, mask, weight = Y[:, 0], mask[:, 0], weight[:, 0]
            elif self._targets_1d and pred.shape[1] == 1:
                pred, y_true, mask, weight = pred[:, 0], Y[:, 0], mask[:, 0], weight[:, 0]
            if self._metric_takes_era(metric):
                return float(metric(y_true, pred, weight, mask, era=ctx.era_index))
            return float(metric(y_true, pred, weight, mask))

        if metric == "logloss":
            return weighted_log_loss_torch(Y[:, 0], pred, sample_weight=ctx.row_weight).item()
        if metric == "accuracy":
            labels = torch.argmax(pred, dim=1)
            return 1.0 - weighted_accuracy_torch(Y[:, 0], labels, sample_weight=ctx.row_weight).item()
        if metric == "mse":
            return weighted_multi_mse_torch(
                Y, pred, sample_weight=target_weight, valid_mask=ctx.target_mask
            ).item()

        valid = ctx.target_mask[:, 0]
        y_v, p_v, w_v = Y[valid, 0], pred[valid, 0], target_weight[valid, 0]
        if metric == "corr":
            return weighted_corr_loss_torch(y_v, p_v, sample_weight=w_v).item()
        if metric == "rmsle":
            return weighted_rmsle_torch(y_v, p_v, sample_weight=w_v).item()
        raise ValueError(f"Invalid eval_metric: {metric}.")

    @staticmethod
    def _metric_takes_era(metric):
        try:
            params = inspect.signature(metric).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            p.name == "era" or p.kind == inspect.Parameter.VAR_KEYWORD for p in params
        )

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
        train_metric = "loss" if obj.has_loss() else "mse"
        train_loss = self._metric_value(train_metric, self.Y_gpu, self.scores, self.ctx)
        self.training_loss.append(train_loss)

        if i % self.eval_every_n_trees == 0:
            eval_loss = self._metric_value(
                self.eval_metric, self.Y_gpu_eval, self._eval_scores, self.ctx_eval
            )
            self.eval_loss.append(eval_loss)

            if len(self.eval_loss) > self.early_stopping_rounds:
                if self.eval_loss[-(self.early_stopping_rounds+1)] < self.eval_loss[-1]:
                    self.stop = True

            metric_name = getattr(self.eval_metric, "__name__", str(self.eval_metric))
            print(
                f"🌲 Tree {i+1}/{self.n_estimators} | Train {train_metric}: {train_loss:.6f} | Eval {metric_name}: {eval_loss:.6f}"
            )

    # ─── Binning / prediction ───

    def bin_data_with_existing_edges(self, X_np):
        num_samples = X_np.shape[0]
        X_float_t = (
            torch.as_tensor(X_np, device=self.device, dtype=torch.float32)
            .transpose(0, 1)
            .contiguous()
        )
        bin_indices_t = torch.empty(
            (self.num_features, num_samples), dtype=torch.int8, device=self.device
        )
        bin_edges = self.bin_edges.to(self.device)
        with torch.no_grad():
            for f in range(self.num_features):
                _get_node_kernel().custom_cuda_binner(
                    X_float_t[f],
                    bin_edges[f].contiguous(),
                    bin_indices_t[f],
                    self.nan_bin,
                )

        return bin_indices_t.transpose(0, 1).contiguous()

    def bin_inference_data(self, X_np):
        is_integer_type = np.issubdtype(X_np.dtype, np.integer)

        if is_integer_type and X_np.shape[1] == self.num_features:
            max_vals = X_np.max(axis=0)
            if np.all(max_vals < self.num_bins):
                print("Detected pre-binned input at predict-time — skipping binning.")
                is_prebinned = True
            else:
                is_prebinned = False
        else:
            is_prebinned = False

        if is_prebinned:
            bin_indices = torch.as_tensor(
                X_np, device=self.device, dtype=torch.int8
            ).contiguous()
        else:
            bin_indices = self.bin_data_with_existing_edges(X_np)
        return bin_indices

    def _predict_raw_binned(self, bin_indices):
        """Raw scores [N, K] for binned data."""
        num_samples = bin_indices.size(0)
        out = (
            torch.zeros(
                (num_samples, self.num_outputs), device=self.device, dtype=torch.float32
            )
            + self._base_score_tensor()
        )
        if self._trees_trained > 0:
            self._accumulate_tree_tensors(
                bin_indices, self._get_cached_tree_tensors(), out
            )
        return out

    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

    def predict_binned(self, bin_indices):
        """Transformed predictions for binned data ([N] for single-output targets)."""
        pred = self.objective_.transform(self._predict_raw_binned(bin_indices))
        if not self.objective_.is_classifier and self._targets_1d and pred.shape[1] == 1:
            return pred[:, 0]
        return pred

    def predict_raw(self, X_np):
        """
        Raw (untransformed) scores, e.g. logits for classification.

        Returns shape (n_samples,) for single-output regression-like models, else (n_samples, K).
        """
        self._check_fitted()
        bin_indices = self.bin_inference_data(X_np)
        raw = self._predict_raw_binned(bin_indices).cpu().numpy()
        del bin_indices
        if not self.objective_.is_classifier and self._targets_1d and raw.shape[1] == 1:
            return raw.reshape(-1)
        return raw

    def predict(self, X_np):
        """
        Predict on new data.

        For regression / custom objectives: returns objective.transform(raw scores)
        For classification: returns predicted class labels
        """
        self._check_fitted()
        if self.objective_.is_classifier:
            # Classification: return class labels
            probs = self.predict_proba(X_np)
            class_indices = np.argmax(probs, axis=1)
            return self.label_encoder.inverse_transform(class_indices)

        bin_indices = self.bin_inference_data(X_np)
        preds = self.predict_binned(bin_indices).cpu().numpy()
        del bin_indices
        return preds

    def predict_proba(self, X_np):
        """
        Predict class probabilities (classification only).

        Returns:
            np.array of shape (n_samples, n_classes) with probabilities
        """
        if not self._is_classifier():
            raise ValueError("predict_proba only available for classification objectives")
        self._check_fitted()

        bin_indices = self.bin_inference_data(X_np)
        probs = self.predict_proba_binned(bin_indices).cpu().numpy()
        del bin_indices
        return probs

    def predict_proba_binned(self, bin_indices):
        """
        Predict probabilities from pre-binned data (classification).

        Returns:
            torch.Tensor of shape (n_samples, n_classes) with probabilities
        """
        return self.objective_.transform(self._predict_raw_binned(bin_indices))

    # ─── Feature importance ───

    def get_feature_importance(self, importance_type='gain', normalize=True):
        """
        Get feature importance scores.

        Parameters:
        -----------
        importance_type : str, default='gain'
            Type of importance. Currently only 'gain' is supported.
        normalize : bool, default=True
            If True, normalize importances to sum to 1.0

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
        if self.feature_importance_ is None:
            raise ValueError("Model has not been fitted yet.")

        if importance_type != 'gain':
            raise ValueError(f"importance_type '{importance_type}' not supported. Use 'gain'.")

        importance = self.feature_importance_.copy()

        if normalize and importance.sum() > 0:
            importance = importance / importance.sum()

        return importance

    def get_per_era_feature_importance(self, normalize=True):
        """
        Get per-era feature importance scores.

        Parameters:
        -----------
        normalize : bool, default=True
            If True, normalize importances within each era to sum to 1.0

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
        if self.per_era_feature_importance_ is None:
            raise ValueError("Model has not been fitted yet.")

        importance = self.per_era_feature_importance_.copy()

        if normalize:
            for era_idx in range(importance.shape[0]):
                era_sum = importance[era_idx].sum()
                if era_sum > 0:
                    importance[era_idx] /= era_sum

        return importance

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
            "era_splitting": self.era_splitting,
            "L2_reg": self.L2_reg,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "warm_start": self.warm_start,
            "monotonic_constraints": self.monotonic_constraints,
            # Trained state
            "objective_": self.objective_,
            "forest": self.forest,
            "bin_edges": to_cpu(self.bin_edges),
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
            "eval_loss": self.eval_loss if hasattr(self, "eval_loss") else [],
            "monotonic_constraints_vector_": self.monotonic_constraints_vector_,
            "monotonic_required_split_direction_": self.monotonic_required_split_direction_,
            "_monotonic_constraints_fitted_": self._monotonic_constraints_fitted_,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

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
        self.era_splitting = state.get('era_splitting', True)
        self.L2_reg = state['L2_reg']
        self.colsample_bytree = state['colsample_bytree']
        self.random_state = state['random_state']
        self.warm_start = state['warm_start']
        self.monotonic_constraints = state.get("monotonic_constraints", None)

        # Restore trained state
        self.objective_ = state.get("objective_") or resolve_objective(self.objective)
        self.forest = state['forest']
        bin_edges = state['bin_edges']
        self.bin_edges = bin_edges.to(self.device) if torch.is_tensor(bin_edges) else bin_edges
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

        self.feature_importance_ = state['feature_importance_']
        self.per_era_feature_importance_ = state['per_era_feature_importance_']
        self._is_fitted = state['_is_fitted']
        self._trees_trained = state['_trees_trained']
        self.training_loss = state.get('training_loss', [])
        self.eval_loss = state.get('eval_loss', [])
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
