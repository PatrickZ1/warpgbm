"""
Compact storage of the boosting rounds of a WarpGBM model.

The trees of a model live in a tuple of immutable ``Block`` objects, each holding consecutive
rounds as flat numpy arrays. The same arrays are what the CPU and GPU predictors read: there is
no separate packed cache. A block is never modified after it is built, so copies, forks and
slices of a model share their blocks (``copy.deepcopy`` returns the block itself); the GPU copy
of a block is cached on the block and therefore shared as well.

Layout of a block with T trees in R rounds (nodes of a tree in pre-order; the left child of node
i is node i + 1, so only the right child is stored):

    feature     [n_nodes] int32   split feature, or -1 - leaf index (within the tree) for a leaf
    word        [n_nodes] int32   split: bin | nan_left << 8 | (right child - i) << 9;
                                  leaf: its training rows ("samples")
    gain        [n_nodes] float32 split gain (NaN when unknown, 0 for leaves)
    values      [n_values] float32 leaf values, NOT multiplied by the learning rate; tree t holds
                [n_leaves, n_cols[t]] values from value_start[t] (row = leaf, column = output)
    node_start, value_start, col_start  [T + 1] int64 per-tree offsets
    n_cols      [T] int32   score columns the tree adds to: num_outputs = all of them in order,
                            0 = none (the tree is skipped), else the columns cols[col_start[t]:]
    cols        int32       column indices of the trees with 0 < n_cols < num_outputs (ascending)
    lr          [T] float32 learning rate of the tree's round
    tree_depth  [T] int8    depth of every tree (0 for a single leaf or an empty tree)
    round_start [R + 1] int64 first tree of every round

A tree only keeps the columns with a non-zero leaf value: a vector tree grown on a few active
outputs stores those, a per-output tree its own column. Dropping the other columns is exact:
the predictors start every sum at +0.0, so a sum is never -0.0, and adding a zero of either
sign leaves it unchanged.
"""

import bisect
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch

# Largest right-child offset of the node word (bits 9 … 30)
_MAX_RIGHT_OFFSET = (1 << 22) - 1
_INT32_LIMIT = 2**31 - 1

_ARRAYS = (
    "feature", "word", "gain", "values", "node_start", "value_start", "col_start",
    "cols", "n_cols", "lr", "tree_depth", "round_start",
)


def _readonly(array):
    array.flags.writeable = False
    return array


def _to_device(array, device):
    """numpy array (also read-only) -> tensor on device, copied asynchronously from pinned memory."""
    array = np.asarray(array)
    host = torch.empty(array.shape, dtype=torch.from_numpy(np.empty(0, array.dtype)).dtype, pin_memory=True)
    host.numpy()[...] = array
    return host.to(device, non_blocking=True)


class Block:
    """Immutable group of consecutive boosting rounds (see the module docstring)."""

    def __init__(self, num_outputs, **arrays):
        self.num_outputs = int(num_outputs)
        for name in _ARRAYS:
            setattr(self, name, _readonly(np.ascontiguousarray(arrays[name])))
        self._device = {}
        self._cpu_plans = {}

    # ─── Shape ───

    @property
    def num_trees(self):
        return self.n_cols.shape[0]

    @property
    def num_rounds(self):
        return self.round_start.shape[0] - 1

    @property
    def nbytes(self):
        """Host bytes of the block's arrays."""
        return sum(getattr(self, name).nbytes for name in _ARRAYS)

    def device_nbytes(self, device=None):
        """Bytes of the cached device copies (of one device, or all)."""
        total = 0
        for key, tensors in self._device.items():
            if device is None or key == _device_key(device):
                total += sum(t.untyped_storage().nbytes() for t in tensors if t is not None)
        return total

    # ─── Immutability, copies, pickling ───

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __getstate__(self):
        state = {name: getattr(self, name) for name in _ARRAYS}
        state["num_outputs"] = self.num_outputs
        return state

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return f"<Block of {self.num_rounds} rounds, {self.num_trees} trees, {self.nbytes} bytes>"

    # ─── Construction ───

    @staticmethod
    def build(pieces, round_sizes, num_outputs):
        """Block from per-tree pieces (see ForestBuilder.add_tree) and trees per round."""
        T = len(pieces)
        node_start = np.zeros(T + 1, dtype=np.int64)
        value_start = np.zeros(T + 1, dtype=np.int64)
        col_start = np.zeros(T + 1, dtype=np.int64)
        n_cols = np.zeros(T, dtype=np.int32)
        for t, p in enumerate(pieces):
            node_start[t + 1] = node_start[t] + p["feature"].shape[0]
            value_start[t + 1] = value_start[t] + p["values"].size
            n_cols[t] = p["n_cols"]
            col_start[t + 1] = col_start[t] + (0 if p["cols"] is None else p["cols"].size)

        def cat(name, dtype):
            parts = [p[name] for p in pieces if p[name] is not None and p[name].size]
            if not parts:
                return np.zeros(0, dtype=dtype)
            return np.concatenate([np.asarray(a, dtype=dtype).reshape(-1) for a in parts])

        return Block(
            num_outputs,
            feature=cat("feature", np.int32),
            word=cat("word", np.int32),
            gain=cat("gain", np.float32),
            values=cat("values", np.float32),
            node_start=node_start,
            value_start=value_start,
            col_start=col_start,
            cols=cat("cols", np.int32),
            n_cols=n_cols,
            lr=np.array([p["lr"] for p in pieces], dtype=np.float32),
            tree_depth=np.array([p["depth"] for p in pieces], dtype=np.int8),
            round_start=np.concatenate([[0], np.cumsum(round_sizes, dtype=np.int64)]).astype(np.int64),
        )

    def slice_rounds(self, start, stop):
        """Block of the rounds [start, stop): the block itself if that is all of it, else a copy."""
        if start == 0 and stop == self.num_rounds:
            return self
        t0, t1 = int(self.round_start[start]), int(self.round_start[stop])
        n0, n1 = int(self.node_start[t0]), int(self.node_start[t1])
        v0, v1 = int(self.value_start[t0]), int(self.value_start[t1])
        c0, c1 = int(self.col_start[t0]), int(self.col_start[t1])
        return Block(
            self.num_outputs,
            feature=self.feature[n0:n1].copy(),
            word=self.word[n0:n1].copy(),
            gain=self.gain[n0:n1].copy(),
            values=self.values[v0:v1].copy(),
            node_start=self.node_start[t0 : t1 + 1] - n0,
            value_start=self.value_start[t0 : t1 + 1] - v0,
            col_start=self.col_start[t0 : t1 + 1] - c0,
            cols=self.cols[c0:c1].copy(),
            n_cols=self.n_cols[t0:t1].copy(),
            lr=self.lr[t0:t1].copy(),
            tree_depth=self.tree_depth[t0:t1].copy(),
            round_start=self.round_start[start : stop + 1] - t0,
        )

    @staticmethod
    def concat(blocks):
        """One block holding the rounds of blocks in order (a new block; the inputs are unchanged)."""
        if len(blocks) == 1:
            return blocks[0]
        arrays = {}
        for name in ("feature", "word", "gain", "values", "cols", "n_cols", "lr", "tree_depth"):
            arrays[name] = np.concatenate([getattr(b, name) for b in blocks])
        for name in ("node_start", "value_start", "col_start", "round_start"):
            parts, offset = [np.zeros(1, dtype=np.int64)], 0
            for b in blocks:
                a = getattr(b, name)
                parts.append(a[1:] + offset)
                offset += int(a[-1])
            arrays[name] = np.concatenate(parts)
        return Block(blocks[0].num_outputs, **arrays)

    # ─── Columns ───

    def tree_columns(self, t):
        """Score columns tree t adds to, as an int array (ascending)."""
        n = int(self.n_cols[t])
        if n == self.num_outputs:
            return np.arange(n)
        c = int(self.col_start[t])
        return self.cols[c : c + n] if n else np.zeros(0, dtype=np.int32)

    def round_columns(self, r):
        """Sorted tuple of the score columns the trees of round r add to."""
        cols = set()
        for t in range(int(self.round_start[r]), int(self.round_start[r + 1])):
            cols.update(int(c) for c in self.tree_columns(t))
        return tuple(sorted(cols))

    def _masks(self):
        """Column bit mask of every tree (num_outputs <= 32)."""
        K = self.num_outputs
        masks = np.zeros(self.num_trees, dtype=np.uint32)
        masks[self.n_cols == K] = np.uint32((1 << K) - 1)
        sparse = (self.n_cols > 0) & (self.n_cols < K)
        if sparse.any():
            counts = np.diff(self.col_start)
            tree_of_col = np.repeat(np.arange(self.num_trees), counts)
            np.bitwise_or.at(masks, tree_of_col, (np.uint32(1) << self.cols.astype(np.uint32)))
        return masks

    # ─── Device copy ───

    def device_arrays(self, device, cache=True):
        """
        (nodes [n, 2] int32, values, meta [T, 4] int32, lr [T], cols or None) on device for the
        CUDA predictor; meta = (first node, first value, column mask (num_outputs <= 32) or
        first column index, n_cols) per tree. Cached on the block unless cache=False.
        """
        key = _device_key(device)
        hit = self._device.get(key)
        if hit is not None:
            return hit
        if int(self.node_start[-1]) > _INT32_LIMIT or int(self.value_start[-1]) > _INT32_LIMIT:
            raise ValueError("A forest block is too large for the CUDA predictor (2^31 nodes or values).")
        K = self.num_outputs
        meta = np.empty((self.num_trees, 4), dtype=np.int32)
        meta[:, 0] = self.node_start[:-1]
        meta[:, 1] = self.value_start[:-1]
        meta[:, 2] = self._masks().view(np.int32) if K <= 32 else self.col_start[:-1]
        meta[:, 3] = self.n_cols
        nodes = np.stack([self.feature, self.word], axis=1)
        cols = _to_device(self.cols, device) if K > 32 and self.cols.size else None
        tensors = (
            _to_device(nodes, device),
            _to_device(self.values, device),
            _to_device(meta, device),
            _to_device(self.lr, device),
            cols,
        )
        if cache:
            self._device[key] = tensors
        return tensors

    def release(self, device=None):
        """Drop the cached device copy (of one device, or all)."""
        if device is None:
            self._device = {}
        else:
            self._device.pop(_device_key(device), None)

    # ─── CPU prediction ───

    def _cpu_plan(self, n_trees):
        """
        _CpuPlan of the first n_trees trees: the trees with columns (t_sel), the columns they
        use, how their values map to those columns, and their roots, value offsets and rates.
        """
        plan = self._cpu_plans.get(n_trees)
        if plan is not None:
            return plan
        K = self.num_outputs
        t_sel = np.flatnonzero(self.n_cols[:n_trees] > 0)
        nc = self.n_cols[t_sel].astype(np.int64)
        if (nc == K).any():
            used = np.arange(K)
        else:
            # Without dense trees, the column lists of the first n_trees trees are all used columns
            used = np.unique(self.cols[: int(self.col_start[n_trees])]).astype(np.int64)
        pos = np.full(K, -1, dtype=np.int64)
        pos[used] = np.arange(used.size)
        groups = []
        for c in np.unique(nc):
            g = np.flatnonzero(nc == c)
            if c == K:
                cpos = np.broadcast_to(pos, (g.size, K))
            else:
                starts = self.col_start[t_sel[g]]
                cpos = pos[self.cols[starts[:, None] + np.arange(c)]]
            groups.append((g, cpos, np.arange(c)))
        # All trees with the same columns in order: values are gathered directly, no scatter
        cpos = groups[0][1] if len(groups) == 1 else None
        direct = (
            cpos is not None
            and cpos.shape[1] == used.size
            and np.array_equal(cpos, np.broadcast_to(np.arange(used.size), cpos.shape))
        )
        index = np.int32 if self.node_start[-1] <= _INT32_LIMIT else np.int64
        plan = _CpuPlan(
            t_sel=t_sel,
            used=used,
            groups=groups,
            direct=direct,
            roots=self.node_start[t_sel].astype(index),
            vstart=self.value_start[t_sel],
            nc=nc,
            lr=self.lr[t_sel],
            depth=int(self.tree_depth[t_sel].max()) if t_sel.size else 0,
            nodes=int(self.node_start[n_trees]),
        )
        # Plans of the whole block are reused by every prediction; prefixes only by the next one
        if n_trees != self.num_trees:
            self._cpu_plans = {k: v for k, v in self._cpu_plans.items() if k == self.num_trees}
        self._cpu_plans[n_trees] = plan
        return plan

    def _leaf_values_cpu(self, plan, leaf):
        """Leaf values x learning rate [rows, trees, used columns] for leaf indices [rows, trees]."""
        voff = plan.vstart + leaf.astype(np.int64) * plan.nc
        if plan.direct:
            vals = self.values.take(voff[:, :, None] + plan.groups[0][2])
        else:
            vals = np.zeros((leaf.shape[0], plan.t_sel.size, plan.used.size), dtype=np.float32)
            for g, cpos, offs in plan.groups:
                vals[:, g[:, None], cpos] = self.values.take(voff[:, g, None] + offs)
        vals *= plan.lr[None, :, None]
        return vals

    def scores_cpu(self, bins, nan_bin, n_trees, carry, ckpts=()):
        """
        Add the trees 0 … n_trees-1 to carry [N, K] (float32 tree sums without the base score)
        on the CPU, for internal int8 bins [N, F]. ckpts: ascending tree counts <= n_trees.
        Returns (sums after n_trees trees, [sums after ckpts[p] trees]), each [N, K] float32.
        The same float32 operations as the CUDA kernel: leaf value x learning rate, added to the
        running sum tree by tree.
        """
        plan = self._cpu_plan(n_trees)
        final = carry.copy()
        at_ckpt = [carry.copy() for _ in ckpts]
        N = bins.shape[0]
        Ts = plan.t_sel.size
        if Ts == 0 or N == 0:
            return final, at_ckpt
        k_ckpt = np.searchsorted(plan.t_sel, np.asarray(ckpts, dtype=np.int64))
        chunk = _row_chunk(Ts, plan.used.size)
        for a in range(0, N, chunk):
            b = min(N, a + chunk)
            node = _traverse(self.feature, self.word, plan.roots, plan.depth, bins[a:b], nan_bin)
            vals = self._leaf_values_cpu(plan, -1 - self.feature.take(node))
            sums, at = _accumulate(vals, carry[a:b], plan.used, k_ckpt)
            final[a:b] = sums
            for out, s in zip(at_ckpt, at):
                out[a:b] = s
        return final, at_ckpt

    # ─── Tree dicts (the model.forest view) ───

    def tree_dict(self, t, scalar_leaf):
        """Tree t as the nested dict of the forest view ({} for an empty tree)."""
        n0, n1 = int(self.node_start[t]), int(self.node_start[t + 1])
        if n0 == n1:
            return {}
        K = self.num_outputs
        feature = self.feature[n0:n1].tolist()
        word = self.word[n0:n1].tolist()
        gain = self.gain[n0:n1].tolist()
        nc = int(self.n_cols[t])
        v0, v1 = int(self.value_start[t]), int(self.value_start[t + 1])
        vals = self.values[v0:v1].reshape(-1, nc).tolist() if nc else None
        cols = self.tree_columns(t).tolist()

        def leaf_value(i):
            if scalar_leaf:
                return float(vals[i][0]) if nc else 0.0
            value = [0.0] * K
            if nc:
                for c, v in zip(cols, vals[i]):
                    value[c] = float(v)
            return value

        def build(i):
            f = feature[i]
            w = word[i]
            if f < 0:
                return {"leaf_value": leaf_value(-1 - f), "samples": w}
            node = {"feature": int(f), "bin": w & 0xFF, "nan_left": bool((w >> 8) & 1)}
            node["left"] = build(i + 1)
            node["right"] = build(i + (w >> 9))
            if gain[i] == gain[i]:  # NaN: saved before split gains were stored
                node["gain"] = float(gain[i])
            return node

        return build(0)


# ─── CPU prediction helpers ───


class _CpuPlan(NamedTuple):
    t_sel: np.ndarray  # trees with columns among the first n_trees
    used: np.ndarray  # score columns those trees add to
    groups: list  # (trees, column positions [trees, c], arange(c)) per column count c
    direct: bool  # all trees add to `used` in order (no scatter)
    roots: np.ndarray  # first node of every selected tree
    vstart: np.ndarray  # first leaf value of every selected tree
    nc: np.ndarray  # columns of every selected tree
    lr: np.ndarray  # learning rate of every selected tree
    depth: int  # deepest selected tree
    nodes: int  # nodes of the first n_trees trees


def _row_chunk(n_trees, n_cols):
    """Rows per chunk that bound the [rows, trees, columns] intermediate arrays."""
    return max(1, (1 << 22) // (max(n_trees, 1) * max(n_cols, 4)))


def _traverse(feature, word, roots, depth, bins, nan_bin):
    """
    Node ids [rows, trees] of the leaves that the rows of internal bins [rows, F] reach from the
    tree roots (node arrays in the Block layout), in one vectorized walk of depth steps.
    """
    rows, F = bins.shape
    flat = bins.reshape(-1).astype(np.int32)
    row_offset = (np.arange(rows, dtype=np.int64) * F)[:, None]
    node = np.broadcast_to(roots, (rows, roots.size)).copy()
    for _ in range(depth):
        f = feature.take(node)
        is_leaf = f < 0
        if is_leaf.all():
            break
        # Leaves (f < 0) read some other bin, clipped into range; their step is 0
        x = flat.take(row_offset + f, mode="clip")
        w = word.take(node)
        go_left = np.where(x == nan_bin, (w & 0x100) != 0, x <= (w & 0xFF))
        node += np.where(is_leaf, 0, np.where(go_left, 1, w >> 9))
    return node


def _accumulate(vals, carry, used, k_ckpt=()):
    """
    (carry + the trees' values added one by one, [the same after k_ckpt[p] trees]) for vals
    [rows, trees, len(used)] (modified in place) and carry [rows, K]. The first addition is
    carry + tree 0, as in the CUDA kernel (also for carry = 0).
    """
    final = carry.copy()
    at_ckpt = [carry.copy() for _ in k_ckpt]
    if vals.shape[1] == 0:
        return final, at_ckpt
    start = carry[:, used]
    vals[:, 0] += start
    np.add.accumulate(vals, axis=1, out=vals)
    final[:, used] = vals[:, -1]
    for out, k in zip(at_ckpt, k_ckpt):
        out[:, used] = vals[:, k - 1] if k > 0 else start
    return final, at_ckpt


def trie_sums_cpu(root, bins, nan_bin, num_outputs, emit):
    """
    Tree sums of the models of a prefix trie of blocks (nodes {"block", "n_trees", "children",
    "models"}) for internal bins [N, F] on the CPU: emit(model, rows slice, sums [rows, K]). All
    trees of all blocks are walked in ONE vectorized traversal per chunk of rows, then every trie
    node adds its trees to the sums of its parent: the same values and additions as
    Block.scores_cpu, so the sums equal each model's own prediction bit for bit.
    """
    need = {}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node["children"].values():
            block = child["block"]
            n = max(need.get(id(block), (block, 0))[1], child["n_trees"])
            need[id(block)] = (block, n)
            stack.append(child)
    parts = [(block, block._cpu_plan(n)) for block, n in need.values()]
    offsets = np.cumsum([0] + [plan.nodes for _, plan in parts], dtype=np.int64)
    feature = np.concatenate([b.feature[: p.nodes] for b, p in parts] or [np.zeros(0, np.int32)])
    word = np.concatenate([b.word[: p.nodes] for b, p in parts] or [np.zeros(0, np.int32)])
    index = np.int32 if offsets[-1] <= _INT32_LIMIT else np.int64
    roots = np.concatenate(
        [p.roots.astype(index) + index(o) for (_, p), o in zip(parts, offsets[:-1])]
        or [np.zeros(0, index)]
    )
    depth = max([p.depth for _, p in parts], default=0)
    N = bins.shape[0]
    chunk = _row_chunk(roots.size, max([p.used.size for _, p in parts], default=1))
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        rows = slice(a, b)
        leaf = -1 - feature.take(_traverse(feature, word, roots, depth, bins[rows], nan_bin))
        vals, col = {}, 0
        for block, plan in parts:
            Ts = plan.t_sel.size
            vals[id(block)] = (block._leaf_values_cpu(plan, leaf[:, col : col + Ts]), plan)
            col += Ts
        zero = np.zeros((b - a, num_outputs), dtype=np.float32)

        def visit(node, sums):
            for i in node["models"]:
                emit(i, rows, sums)
            for child in node["children"].values():
                v, plan = vals[id(child["block"])]
                k = int(np.searchsorted(plan.t_sel, child["n_trees"]))
                visit(child, _accumulate(v[:, :k].copy(), sums, plan.used)[0])

        visit(root, zero)


def _device_key(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return str(device)


class ForestBuilder:
    """Collects the trees of the rounds grown by a fit; build() turns them into a Block."""

    def __init__(self, num_outputs):
        self.num_outputs = int(num_outputs)
        self._pieces = []
        self._round_sizes = []
        self._open = 0  # trees of the round being added

    @property
    def num_rounds(self):
        return len(self._round_sizes)

    def add_tree(self, feature, word, gain, leaf, cols, lr, depth):
        """
        Add a tree: node arrays in the Block layout, leaf values [n_leaves, len(cols)] for the
        score columns cols (None: all outputs), the learning rate of its round and its depth.
        """
        K = self.num_outputs
        leaf = np.asarray(leaf, dtype=np.float32)
        leaf = leaf.reshape(leaf.shape[0], -1) if leaf.size else np.zeros((0, 0), dtype=np.float32)
        cols = np.arange(K) if cols is None else np.asarray(cols, dtype=np.int64).reshape(-1)
        if leaf.shape[0] and leaf.shape[1] != cols.size:
            raise ValueError(f"leaf values have {leaf.shape[1]} columns for {cols.size} score columns")
        if cols.size > 1 and (np.diff(cols) <= 0).any():
            order = np.argsort(cols, kind="stable")
            cols, leaf = cols[order], leaf[:, order]
        # Columns whose leaves are all zero add nothing (exact, see the module docstring)
        keep = (leaf != 0).any(axis=0) if leaf.shape[0] else np.zeros(cols.size, dtype=bool)
        if not keep.all():
            cols, leaf = cols[keep], leaf[:, keep]
        n_cols = cols.size
        dense = n_cols == K
        feature = np.asarray(feature, dtype=np.int32)
        if feature.size > _MAX_RIGHT_OFFSET:
            raise ValueError("Trees with more than 2^22 nodes are not supported.")
        self._pieces.append(
            {
                "feature": feature,
                "word": np.asarray(word, dtype=np.int32),
                "gain": np.asarray(gain, dtype=np.float32),
                "values": np.ascontiguousarray(leaf).reshape(-1),
                "n_cols": n_cols,
                "cols": None if dense or n_cols == 0 else cols.astype(np.int32),
                "lr": np.float32(lr),
                "depth": int(depth),
            }
        )
        self._open += 1

    def end_round(self):
        self._round_sizes.append(self._open)
        self._open = 0

    def round_block(self, r=-1):
        """Block of round r of this builder only (e.g. the newest round, for evaluation)."""
        r = range(self.num_rounds)[r]
        start = sum(self._round_sizes[:r])
        size = self._round_sizes[r]
        return Block.build(self._pieces[start : start + size], [size], self.num_outputs)

    def build(self):
        """Block of all complete rounds (None if there are none)."""
        if not self._round_sizes:
            return None
        n = sum(self._round_sizes)
        return Block.build(self._pieces[:n], self._round_sizes, self.num_outputs)


# ─── Tree dicts <-> blocks ───


def tree_arrays_from_dict(tree, leaf_width):
    """
    (feature, word, gain, leaf [n_leaves, leaf_width], depth) of a tree dict, with the leaf
    rules of the v2.8 packer: a list leaf value is cut to leaf_width, a scalar fills it.
    """
    feature, word, gain, leaves = [], [], [], []
    depth = 0
    stack = [(tree, -1, 0)] if tree else []  # (node, node to patch with its right offset, depth)
    while stack:
        node, patch, d = stack.pop()
        i = len(feature)
        if patch >= 0:
            word[patch] |= (i - patch) << 9
        if "leaf_value" in node:
            value = node["leaf_value"]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            row = np.zeros(leaf_width, dtype=np.float32)
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value, dtype=np.float32).reshape(-1)[:leaf_width]
                row[:] = arr if arr.size == leaf_width else arr[0]
            else:
                row[:] = value
            feature.append(-1 - len(leaves))
            word.append(int(node.get("samples", 0)))
            gain.append(0.0)
            leaves.append(row)
            depth = max(depth, d)
        else:
            feature.append(int(node["feature"]))
            word.append(int(node["bin"]) | (int(bool(node.get("nan_left", False))) << 8))
            gain.append(float(node["gain"]) if "gain" in node else float("nan"))
            stack.append((node["right"], i, d + 1))
            stack.append((node["left"], -1, d + 1))
    leaf = np.stack(leaves) if leaves else np.zeros((0, leaf_width), dtype=np.float32)
    return feature, word, gain, leaf, depth


def blocks_from_dicts(rounds, learning_rates, num_outputs, per_output):
    """Blocks of a forest of tree dicts (list of rounds; per-output rounds are lists of K trees)."""
    builder = ForestBuilder(num_outputs)
    leaf_width = 1 if per_output else num_outputs
    for round_trees, lr in zip(rounds, learning_rates):
        trees = round_trees if per_output else [round_trees]
        for k, tree in enumerate(trees):
            feature, word, gain, leaf, depth = tree_arrays_from_dict(tree, leaf_width)
            cols = [k] if per_output else None
            builder.add_tree(feature, word, gain, leaf, cols, lr, depth)
        builder.end_round()
    block = builder.build()
    return () if block is None else (block,)


def blocks_prefix(blocks, num_rounds):
    """The blocks holding the first num_rounds rounds (the last one cut if needed)."""
    return blocks_range(blocks, 0, num_rounds)


def blocks_range(blocks, start, stop):
    """The blocks holding the rounds [start, stop): whole blocks are shared, partial ones copied."""
    out, offset = [], 0
    for b in blocks:
        lo, hi = max(start - offset, 0), min(stop - offset, b.num_rounds)
        if lo < hi:
            out.append(b.slice_rounds(lo, hi))
        offset += b.num_rounds
    return tuple(out)


def num_rounds(blocks):
    return sum(b.num_rounds for b in blocks)


class ForestView(Sequence):
    """
    Read-only view of a model's rounds as tree dicts (the format of WarpGBM.forest before v2.9):
    forest[r] is the tree of round r (a list of K trees for per-output objectives), built on
    access. A split node is {"feature", "bin", "nan_left", "left", "right", "gain"}, a leaf
    {"leaf_value", "samples"} with a float value for single-output and per-output trees and a
    list of K floats otherwise. Editing the returned dicts does not change the model.
    """

    def __init__(self, blocks, num_outputs, per_output):
        self._blocks = tuple(blocks)
        self._per_output = bool(per_output)
        self._scalar = self._per_output or num_outputs == 1
        self._ends = list(np.cumsum([b.num_rounds for b in self._blocks], dtype=np.int64))

    def __len__(self):
        return int(self._ends[-1]) if self._ends else 0

    def __getitem__(self, r):
        if isinstance(r, slice):
            return [self[i] for i in range(*r.indices(len(self)))]
        n = len(self)
        if r < 0:
            r += n
        if not 0 <= r < n:
            raise IndexError("forest round index out of range")
        b = bisect.bisect_right(self._ends, r)
        block = self._blocks[b]
        local = r - (int(self._ends[b - 1]) if b else 0)
        t0, t1 = int(block.round_start[local]), int(block.round_start[local + 1])
        trees = [block.tree_dict(t, self._scalar) for t in range(t0, t1)]
        if self._per_output:
            return trees
        return trees[0] if trees else {}

    def __eq__(self, other):
        if not isinstance(other, (ForestView, list, tuple)):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    __hash__ = None

    def __repr__(self):
        return f"<WarpGBM forest view: {len(self)} rounds>"
