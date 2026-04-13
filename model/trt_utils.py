# Copyright (c) 2025-2026, ETH Zurich (Robotic Systems Lab) & NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Optimise the SAM2 Hiera image encoder (TensorRT) and mask decoder (torch.compile).

Encoder — TensorRT via torch-tensorrt
  Approach follows the official PyTorch-TensorRT SAM2 tutorial:
    https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/torch_export_sam2.py
  Engine cached to ``_TRT_CACHE_DIR``; ~5–10 min first-run, instant on subsequent starts.

Decoder — torch.compile (default)
  TRT compilation of the SAM2 mask decoder fails in TRT 2.5: the TwoWayTransformer's
  cross-attention has a dynamic batch dim (vocab_size × num_tokens) on the query side
  while image-feature key/value tensors start at batch=1.  TRT fuses the resulting
  implicit ``expand`` broadcast with surrounding SHUFFLE layers into a ``ForeignNode``
  for which no valid kernel exists.  The official torch-tensorrt SAM2 tutorial also
  compiles only the image encoder.  ``torch.compile(mode="default")`` achieves good
  per-frame latency (~23 ms vs ~52 ms baseline) while keeping recompilation for new
  vocabulary sizes cheap (~5–10 ms vs ~200 ms for max-autotune).  This avoids latency
  spikes when ``--lmm_per_image`` generates vocabulary sizes outside the warmup range.

Patches (applied by monkey-patching live module instances; no source files modified):
  Encoder:
    1. ``FpnNeck.forward`` — remove forced float32 cast in F.interpolate.
    2. ``LayerNorm2d.forward`` — replace manual mean/var with ``F.layer_norm``.
    3. ``PositionEmbeddingRandom.forward`` — use model dtype for the grid.
  Decoder (also improve torch.compile correctness):
    4. ``Attention.forward`` — remove non-traceable ``sdp_kernel`` context manager;
       inline head-separation using last-dim ops.
    5. ``TwoWayAttentionBlock.forward`` — explicit ``expand`` before cross-attention.
    6. ``MaskDecoder.predict_masks`` — replace ``repeat_interleave``/``view(b,...)``
       with ``expand``/``unflatten``/``flatten``.
    7. ``LayerNorm2d.forward`` — use ``F.layer_norm`` (removes unsqueeze patterns).
"""

import os
import types
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_TRT_CACHE_DIR = os.environ.get("OWSAM_TRT_CACHE", "/app/trt_cache")


def _triton_available() -> bool:
    """Return True if Triton is installed AND can find the CUDA headers it needs to compile.

    Triton imports successfully even when ``cuda.h`` is missing from its include search
    path, but then fails at JIT-compile time with a CalledProcessError.  We check for
    ``cuda.h`` upfront so that ``_torch_compile`` can fall back to ``cudagraphs`` before
    any compilation attempt is made.
    """
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    # Verify the CUDA tools that Triton needs to compile kernels are accessible.
    import shutil
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    # 1. cuda.h — needed by Triton's cuda_utils.c gcc compilation step.
    cuda_h_candidates = [
        os.path.join(cuda_home, "include", "cuda.h"),
        "/usr/local/cuda/include/cuda.h",
        "/usr/include/cuda.h",
    ]
    if not any(os.path.isfile(p) for p in cuda_h_candidates):
        return False
    # 2. ptxas — PTX assembler called by Triton to produce GPU binaries.
    # Triton checks TRITON_PTXAS_PATH first, then falls back to PATH search.
    # On aarch64 JetPack its PATH search is unreliable, so auto-set the env var
    # here when it is absent — Triton reads os.environ at compile time.
    ptxas_path = (
        os.environ.get("TRITON_PTXAS_PATH")
        or shutil.which("ptxas")
        or os.path.join(cuda_home, "bin", "ptxas")
    )
    if not os.path.isfile(ptxas_path):
        return False
    os.environ.setdefault("TRITON_PTXAS_PATH", ptxas_path)
    return True


import threading as _threading_mod


class _PicklableRLock:
    """A ``threading.RLock`` wrapper that is picklable by ANY pickler, including
    torch's internal Pickler subclass.

    Root cause of ``TypeError: cannot pickle '_thread.RLock' object``:

    When ``torch.compile(backend='inductor')`` runs its first forward pass, the
    inductor backend serialises the compiled FX graph's ``constants`` dict (which
    contains objects captured as constants during dynamo tracing — including any
    ``_thread.RLock`` objects stored as module attributes) to write compilation
    artefacts to a temp directory and/or send them to worker processes via pickle.

    ``copyreg.dispatch_table`` is checked FIRST in Python's standard ``Pickler``,
    so ``copyreg.pickle(_thread.RLock, ...)`` would normally work — but torch's
    internal Pickler subclass supplies its own ``dispatch_table`` that does NOT
    include the copyreg entries, so the registration is silently ignored.

    Python's pickle protocol ALWAYS calls ``obj.__reduce_ex__()`` on the object
    itself as a fallback; this step cannot be bypassed by a custom Pickler.
    Replacing the actual ``_thread.RLock`` objects with instances of this wrapper
    class (which defines ``__reduce__``) therefore works regardless of which
    Pickler torch uses internally.

    All threading semantics (acquire / release / context-manager) are fully
    preserved by delegating to an underlying ``_thread.RLock``.
    """

    __slots__ = ("_lock",)

    def __init__(self):
        self._lock = _threading_mod.RLock()

    # Pickle support — produce a fresh unlocked instance on unpickle.
    # torch only pickles for cache-key purposes and never restores lock state.
    def __reduce__(self):
        return (type(self), ())

    def acquire(self, blocking=True, timeout=-1):
        return self._lock.acquire(blocking=blocking, timeout=timeout)

    def release(self):
        return self._lock.release()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def _is_owned(self):
        # Used internally by threading.Condition
        return self._lock._is_owned()


def _replace_rlocks_in_module(module: torch.nn.Module) -> None:
    """Recursively replace every ``_thread.RLock`` in *module* with a
    ``_PicklableRLock`` wrapper.

    Inductor captures module attributes (including locks used for thread-safe
    caching in BEiT-3 / torchscale) as constants in the compiled FX graph and
    then pickles them when writing compilation artefacts.  Replacing the native
    C-extension RLock objects with Python-level wrappers that implement
    ``__reduce__`` makes them picklable by any Pickler.

    The traversal covers:
      * Direct ``__dict__`` attributes of every ``nn.Module`` instance.
      * Non-Module sub-objects reachable from those dicts (e.g. cache helpers).
      * Lists of objects that may contain locks.
    """
    import _thread as _t
    _rlock_type = type(_t.RLock())  # _thread.RLock (same as type(threading.RLock()))
    visited: set = set()

    def _process(obj) -> None:
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)
        d = getattr(obj, "__dict__", None)
        if not d:
            return
        for key, val in list(d.items()):
            if isinstance(val, _rlock_type):
                try:
                    object.__setattr__(obj, key, _PicklableRLock())
                except (AttributeError, TypeError):
                    pass
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, _rlock_type):
                        val[i] = _PicklableRLock()
                    elif hasattr(item, "__dict__") and not isinstance(item, type):
                        _process(item)
            elif hasattr(val, "__dict__") and not isinstance(val, type):
                _process(val)

    for submod in module.modules():
        _process(submod)
    _process(module)  # catch non-Module sub-objects from the top level too


def _make_locks_picklable() -> None:
    """Register copyreg handlers for threading locks (belt-and-suspenders).

    This is kept alongside ``_replace_rlocks_in_module`` for Pickler paths that
    DO respect ``copyreg.dispatch_table``.  It has no effect on torch's internal
    Pickler subclass (which supplies its own dispatch_table), so it is NOT
    sufficient on its own — use ``_replace_rlocks_in_module`` as the primary fix.
    """
    import copyreg
    import _thread

    for lock_type, factory in [
        (_thread.RLock, _thread.RLock),
        (type(_threading_mod.Lock()), _threading_mod.Lock),
    ]:
        try:
            copyreg.pickle(lock_type, lambda obj, f=factory: (f, ()))
        except Exception:
            pass


class _InductorWithCudagraphsFallback(torch.nn.Module):
    """Wrap a torch.compile(inductor) module; fall back to cudagraphs on the first
    forward pass if inductor raises a pickle error.

    torch.compile() is lazy: the returned object is a thin wrapper that triggers
    actual kernel compilation during the FIRST forward call.  If that compilation
    fails (e.g. because threading.RLock objects inside BEiT-3 / torchscale are
    not picklable by torch's internal Pickler), the error surfaces at inference
    time — not at the torch.compile() call site, so a try/except around
    torch.compile() cannot catch it.

    This wrapper catches the TypeError on the first call, recompiles with the
    cudagraphs backend (no pickling required), and continues transparently.
    """

    def __init__(self, compiled: torch.nn.Module, original: torch.nn.Module, fullgraph: bool):
        super().__init__()
        self._compiled = compiled
        self._original = original
        self._fullgraph = fullgraph
        self._failed = False

    def forward(self, *args, **kwargs):
        if not self._failed:
            try:
                return self._compiled(*args, **kwargs)
            except TypeError as exc:
                if "pickle" in str(exc).lower() or "RLock" in str(exc):
                    print(
                        f"[compile] inductor pickle error on first inference ({exc}); "
                        "falling back to cudagraphs",
                        flush=True,
                    )
                    self._failed = True
                    torch._dynamo.reset()
                    try:
                        self._compiled = torch.compile(
                            self._original, backend="cudagraphs", fullgraph=self._fullgraph
                        )
                        print("[compile] fallback to torch.compile(cudagraphs) succeeded", flush=True)
                    except Exception as e2:
                        print(f"[compile] cudagraphs fallback also failed ({e2}); running eager", flush=True)
                        self._compiled = self._original
                else:
                    raise
        return self._compiled(*args, **kwargs)

    # Proxy attribute access to the wrapped compiled module so that
    # downstream code that reads e.g. module.conv_s0 still works.
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._compiled, name)


def _torch_compile(module: torch.nn.Module, *, mode: str = "default", fullgraph: bool = False) -> torch.nn.Module:
    """Compile ``module`` with the best available backend.

    On Jetson aarch64 with PyTorch ≥ 2.6, inductor's Triton kernel compilation
    uses worker *processes* (not threads) for parallelism.  CUDA + ``fork`` is
    unsafe on aarch64, so PyTorch forces the ``spawn`` start method, which
    requires pickling every compilation task including all constants captured from
    the FX graph during dynamo tracing.  ``_thread.RLock`` objects inside BEiT-3
    / torchscale (in ``lru_cache`` wrappers, closures, and class-level attributes)
    appear as captured constants and cannot be pickled — causing
    ``TypeError: cannot pickle '_thread.RLock' object`` at the FIRST forward pass
    (``torch.compile()`` is lazy; compilation happens at inference time).

    On PyTorch 2.5.1 x86 (the desktop server build), the worker pool uses
    ``fork`` and no serialisation is required, so the error never appears there.

    Fix: ``compile_threads=1`` forces single-threaded in-process Triton
    compilation — no subprocess, no IPC, no pickle.  The RLock problem is
    sidestepped entirely; no need to traverse and replace every lock hidden in
    closures or ``lru_cache`` wrappers.  Serial compilation only affects the
    first inference call; steady-state throughput is unchanged.

    Belt-and-suspenders: disk caches are also disabled, and the returned module
    is wrapped in ``_InductorWithCudagraphsFallback`` to transparently fall back
    to ``cudagraphs`` if any pickle path we missed still fires.

    Priority:
      1. Jetson + Triton → ``inductor`` (compile_threads=1, dynamic=True, caches off).
      2. Jetson, inductor still fails → ``cudagraphs`` (via ``_InductorWithCudagraphsFallback``).
      3. Non-Jetson, Triton available → ``inductor`` default (best kernel fusion).
      4. Non-Jetson, Triton unavailable → ``cudagraphs``.
      5. Eager fallback if all backends fail.
    """
    _on_jetson = os.path.isfile("/etc/nv_tegra_release")
    if _on_jetson and _triton_available():
        # torch.compile() is lazy: the wrapper is created immediately but actual
        # kernel compilation happens on the FIRST forward pass.  At that point,
        # inductor serialises the FX graph's constants dict (which contains any
        # _thread.RLock objects captured during dynamo tracing) to write kernel
        # artefacts to a temp dir and send tasks to worker processes.
        # The primary fix is _replace_rlocks_in_module (above): replace the C-
        # extension RLock objects with Python wrappers that define __reduce__ so
        # they are picklable by any Pickler.  Cache disabling below is belt-and-
        # suspenders to reduce the total amount of pickling that happens.
        # dynamic=True compiles one symbolic graph for all vocabulary sizes,
        # avoiding per-shape recompilation overhead.
        # Root cause: on Jetson aarch64 with PyTorch ≥ 2.6, inductor spawns
        # worker PROCESSES (not threads) for parallel Triton kernel compilation.
        # CUDA + fork is unsafe on aarch64, so PyTorch uses `spawn`, which
        # requires pickling the entire compilation task — including all constants
        # captured from the FX graph during dynamo tracing.  BEiT-3 / torchscale
        # store _thread.RLock objects in lru_cache wrappers, closures, and class-
        # level attributes that appear as captured constants and cannot be pickled.
        #
        # On PyTorch 2.5.1 x86 (the desktop server build) the worker pool uses
        # `fork` so no serialisation is needed, which is why the error only
        # appears on Jetson.
        #
        # Fix: compile_threads=1 forces single-threaded in-process Triton
        # compilation.  No subprocess → no IPC → no pickle.  This sidesteps
        # the RLock problem entirely without needing to find and replace every
        # lock hidden in closures, lru_cache wrappers, or class attributes.
        # Compilation is serial, but that only affects the first inference call.
        try:
            import torch._inductor.config as _ic
            _ic.compile_threads = 1          # in-process Triton JIT, no pickle
            _ic.fx_graph_cache = False       # no disk cache serialisation
            if hasattr(_ic, "force_disable_caches"):
                _ic.force_disable_caches = True
            if hasattr(_ic, "autotune_local_cache"):
                _ic.autotune_local_cache = False
        except Exception:
            pass
        try:
            import torch._functorch.config as _fc
            if hasattr(_fc, "enable_autograd_cache"):
                _fc.enable_autograd_cache = False
        except Exception:
            pass
        try:
            compiled = torch.compile(module, mode=mode, fullgraph=fullgraph, dynamic=True)
            print("[compile] Jetson+Triton: using torch.compile(inductor, dynamic=True, single-thread)", flush=True)
            # Wrap in a fallback shim: if the FIRST forward pass raises a pickle
            # error (compilation is lazy, error surfaces at inference time), we
            # transparently switch to the cudagraphs backend instead of crashing.
            return _InductorWithCudagraphsFallback(compiled, module, fullgraph)
        except Exception as e:
            print(f"[compile] inductor failed ({e}); trying cudagraphs", flush=True)
    if _on_jetson:
        try:
            compiled = torch.compile(module, backend="cudagraphs", fullgraph=fullgraph)
            print("[compile] Jetson: using torch.compile(cudagraphs)", flush=True)
            return compiled
        except Exception as e:
            print(f"[compile] cudagraphs failed ({e}); running in eager mode", flush=True)
            return module
    if _triton_available():
        return torch.compile(module, mode=mode, fullgraph=fullgraph)
    # Non-Jetson but Triton unavailable — try cudagraphs before giving up.
    try:
        compiled = torch.compile(module, backend="cudagraphs", fullgraph=fullgraph)
        print("[compile] Triton unavailable; using torch.compile(cudagraphs) fallback", flush=True)
        return compiled
    except Exception as e:
        print(f"[compile] cudagraphs backend failed ({e}); running in eager mode")
        return module


# ---------------------------------------------------------------------------
# Patched forward implementations
# ---------------------------------------------------------------------------

def _fpn_neck_forward_patched(self, xs):
    """FpnNeck.forward without forced float32 cast in F.interpolate."""
    out = [None] * len(self.convs)
    pos = [None] * len(self.convs)
    prev_features = None
    n = len(self.convs) - 1
    for i in range(n, -1, -1):
        x = xs[i]
        lateral_features = self.convs[n - i](x)
        if i in self.fpn_top_down_levels and prev_features is not None:
            top_down_features = F.interpolate(
                prev_features,  # removed .to(dtype=torch.float32)
                scale_factor=2.0,
                mode=self.fpn_interp_model,
                align_corners=(None if self.fpn_interp_model == "nearest" else False),
                antialias=False,
            )
            prev_features = lateral_features + top_down_features
            if self.fuse_type == "avg":
                prev_features = prev_features / 2
        else:
            prev_features = lateral_features
        x_out = prev_features
        out[i] = x_out
        pos[i] = self.position_encoding(x_out).to(x_out.dtype)
    return out, pos


def _layer_norm_2d_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    """LayerNorm2d.forward using F.layer_norm instead of manual mean/var."""
    x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    x = F.layer_norm(
        x,
        normalized_shape=(self.num_channels,),
        weight=self.weight,
        bias=self.bias,
        eps=self.eps,
    )
    x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    return x


def _pos_embed_random_forward_patched(self, size):
    """PositionEmbeddingRandom.forward using model dtype for the grid.

    Preserves the original return shape (C x H x W) — only fixes the forced
    float32 grid creation so no unwanted dtype cast is traced.
    """
    h, w = size
    device = self.positional_encoding_gaussian_matrix.device
    dtype = self.positional_encoding_gaussian_matrix.dtype
    grid = torch.ones((h, w), device=device, dtype=dtype)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w
    pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
    return pe.permute(2, 0, 1)  # C x H x W (same as original)


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _patch_layer_norm_2d(module: nn.Module):
    """Recursively replace LayerNorm2d.forward with the F.layer_norm version."""
    # Import lazily to avoid issues when called before sam2 is on sys.path
    try:
        from model.segment_anything_2.sam2.modeling.sam2_utils import LayerNorm2d
    except ImportError:
        try:
            from sam2.modeling.sam2_utils import LayerNorm2d
        except ImportError:
            LayerNorm2d = None

    patched = 0
    for mod in module.modules():
        if LayerNorm2d is not None and isinstance(mod, LayerNorm2d):
            # Ensure the num_channels attribute exists (added by the torch-trt fork)
            if not hasattr(mod, "num_channels"):
                # weight shape is (num_channels,)
                mod.num_channels = mod.weight.shape[0]
            mod.forward = types.MethodType(_layer_norm_2d_forward_patched, mod)
            patched += 1
    return patched


def _patch_fpn_neck(module: nn.Module):
    """Patch FpnNeck instances inside the encoder."""
    try:
        from model.segment_anything_2.sam2.modeling.backbones.image_encoder import FpnNeck
    except ImportError:
        try:
            from sam2.modeling.backbones.image_encoder import FpnNeck
        except ImportError:
            FpnNeck = None

    patched = 0
    for mod in module.modules():
        if FpnNeck is not None and isinstance(mod, FpnNeck):
            mod.forward = types.MethodType(_fpn_neck_forward_patched, mod)
            patched += 1
    return patched


def _patch_pos_embed_random(module: nn.Module):
    """Patch PositionEmbeddingRandom instances inside the encoder."""
    try:
        from model.segment_anything_2.sam2.modeling.position_encoding import PositionEmbeddingRandom
    except ImportError:
        try:
            from sam2.modeling.position_encoding import PositionEmbeddingRandom
        except ImportError:
            PositionEmbeddingRandom = None

    patched = 0
    for mod in module.modules():
        if PositionEmbeddingRandom is not None and isinstance(mod, PositionEmbeddingRandom):
            mod.forward = types.MethodType(_pos_embed_random_forward_patched, mod)
            patched += 1
    return patched


def _apply_trt_patches(image_encoder: nn.Module) -> None:
    """Apply all patches required for torch.export + TRT compilation (encoder)."""
    n_ln = _patch_layer_norm_2d(image_encoder)
    n_fpn = _patch_fpn_neck(image_encoder)
    n_pe = _patch_pos_embed_random(image_encoder)
    print(f"[TRT] Encoder patches applied: LayerNorm2d×{n_ln}, FpnNeck×{n_fpn}, PosEmbedRandom×{n_pe}")


# ---------------------------------------------------------------------------
# Decoder patches — removes non-traceable context manager and dynamic-b reshapes
# ---------------------------------------------------------------------------

def _attention_forward_patched(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention.forward without the non-traceable sdp_kernel context manager.

    Inlines _separate_heads / _recombine_heads using unflatten+transpose / transpose+flatten
    instead of shape-extraction + reshape.  The original pattern
        b, n, c = x.shape; x.reshape(b, n, num_heads, c // num_heads)
    causes TRT to fuse the ops into a ForeignNode with no valid kernel when the
    batch dim (b = vocab_size) is a dynamic symbolic dimension.  Using only
    last-dim operations avoids this.
    """
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)
    head_dim = q.shape[-1] // self.num_heads
    # separate heads: [B, N, C] -> [B, heads, N, head_dim]
    q = q.unflatten(-1, [self.num_heads, head_dim]).transpose(1, 2)
    k = k.unflatten(-1, [self.num_heads, head_dim]).transpose(1, 2)
    v = v.unflatten(-1, [self.num_heads, head_dim]).transpose(1, 2)
    dropout_p = self.dropout_p if self.training else 0.0
    # Drop the non-traceable context manager; PyTorch will auto-select the best kernel.
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    # recombine heads: [B, heads, N, head_dim] -> [B, N, C]
    out = out.transpose(1, 2).flatten(-2)
    out = self.out_proj(out)
    return out


def _patch_attention(module: nn.Module) -> int:
    """Patch base Attention modules in the decoder (not RoPEAttention)."""
    try:
        from model.segment_anything_2.sam2.modeling.sam.transformer import Attention, RoPEAttention
    except ImportError:
        try:
            from sam2.modeling.sam.transformer import Attention, RoPEAttention
        except ImportError:
            return 0

    patched = 0
    for mod in module.modules():
        # Only patch the base Attention class, not RoPEAttention (used in image encoder, already handled)
        if type(mod) is Attention:
            mod.forward = types.MethodType(_attention_forward_patched, mod)
            patched += 1
    return patched


# ---------------------------------------------------------------------------
# TwoWayAttentionBlock patch — explicit cross-attention expansion
# ---------------------------------------------------------------------------

def _two_way_attn_block_forward_patched(
    self,
    queries: torch.Tensor,
    keys: torch.Tensor,
    query_pe: torch.Tensor,
    key_pe: torch.Tensor,
) -> tuple:
    """TwoWayAttentionBlock.forward with explicit batch-dim expansion before cross-attention.

    In the first block, ``keys`` (image embedding) has batch=1 while ``queries`` (text
    tokens) has batch=N (dynamic, vocab × num_tokens).  TRT's ``scaled_dot_product_attention``
    back-end handles the broadcast implicitly, but it fuses the resulting expand with
    surrounding SHUFFLE layers into a ForeignNode with no valid kernel.

    The fix: expand image features to N **before** passing them to the cross-attention
    modules so TRT sees a plain ``expand`` → no implicit broadcast inside SDPA.
    """
    N = queries.shape[0]  # dynamic

    # Self attention block
    if self.skip_first_layer_pe:
        queries = self.self_attn(q=queries, k=queries, v=queries)
    else:
        q = queries + query_pe
        attn_out = self.self_attn(q=q, k=q, v=queries)
        queries = queries + attn_out
    queries = self.norm1(queries)

    # Cross attention block, tokens attending to image embedding
    # Expand image features to N to avoid implicit SDPA broadcast → TRT ForeignNode
    keys_n = keys.expand(N, -1, -1)
    key_pe_n = key_pe.expand(N, -1, -1)
    q = queries + query_pe
    k = keys_n + key_pe_n
    attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys_n)
    queries = queries + attn_out
    queries = self.norm2(queries)

    # MLP block
    mlp_out = self.mlp(queries)
    queries = queries + mlp_out
    queries = self.norm3(queries)

    # Cross attention block, image embedding attending to tokens
    # After the token→image block, keys_n is [N,4096,256]; expand to match queries.
    # Use keys_n so this block always sees [N,...] rather than [1,...].
    k = keys_n + key_pe_n
    attn_out = self.cross_attn_image_to_token(q=k, k=queries + query_pe, v=queries)
    keys = keys + attn_out
    keys = self.norm4(keys)

    return queries, keys


def _patch_two_way_attn_blocks(module: nn.Module) -> int:
    """Patch TwoWayAttentionBlock instances to use explicit cross-attention expansion."""
    try:
        from model.segment_anything_2.sam2.modeling.sam.transformer import TwoWayAttentionBlock
    except ImportError:
        try:
            from sam2.modeling.sam.transformer import TwoWayAttentionBlock
        except ImportError:
            return 0

    patched = 0
    for mod in module.modules():
        if isinstance(mod, TwoWayAttentionBlock):
            mod.forward = types.MethodType(_two_way_attn_block_forward_patched, mod)
            patched += 1
    return patched


def _predict_masks_patched(
    self,
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    repeat_image: bool,
    high_res_features=None,
):
    """MaskDecoder.predict_masks with TRT-friendly reshape ops.

    Three patterns replaced to avoid TRT ForeignNode errors:

    1. ``torch.repeat_interleave(x, N, dim=0)`` (dynamic N)
       →  ``x.expand(N, -1, -1, -1)``
       TRT decomposes repeat_interleave into reshape+expand+reshape; the final
       reshape references the dynamic batch dim and produces a ForeignNode.
       ``expand`` is a single broadcast op that TRT handles natively.

    2. ``src.transpose(1, 2).view(b, c, h, w)``  →  ``src.transpose(1, 2).unflatten(-1, [h, w])``
       The last dimension is always h*w (fixed); unflatten only splits that dim
       without referencing the dynamic batch dimension b.

    3. ``upscaled_embedding.view(b, c, h*w)`` / ``.view(b, -1, h, w)``
       →  ``upscaled_embedding.flatten(-2)`` / ``.unflatten(-1, [h, w])``
       Same principle: only last-dim operations, no dynamic-b capture.
    """
    s = 0
    if self.pred_obj_scores:
        output_tokens = torch.cat(
            [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        s = 1
    else:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    # FIX: expand instead of repeat_interleave — TRT handles broadcast natively
    # without decomposing into reshape+tile+reshape, which fails with dynamic N.
    N = tokens.shape[0]  # dynamic
    src = image_embeddings.expand(N, -1, -1, -1)
    src = src + dense_prompt_embeddings
    pos_src = image_pe.expand(N, -1, -1, -1)
    _, c, h, w = image_embeddings.shape  # use static shape; b=1 always

    hs, src = self.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, s, :]
    mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

    # FIX: unflatten(-1, [h, w]) — only reshapes the last dim, never touches dynamic b
    src = src.transpose(1, 2).unflatten(-1, [h, w])
    if not self.use_high_res_features:
        upscaled_embedding = self.output_upscaling(src)
    else:
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

    hyper_in_list = []
    for i in range(self.num_mask_tokens):
        hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
    hyper_in = torch.stack(hyper_in_list, dim=1)

    _, c, h, w = upscaled_embedding.shape  # b is dynamic — do not capture it
    # FIX: flatten(-2) merges last two dims without touching b; unflatten(-1,...) restores spatial
    masks = (hyper_in @ upscaled_embedding.flatten(-2)).unflatten(-1, [h, w])

    iou_pred = self.iou_prediction_head(iou_token_out)
    if self.pred_obj_scores:
        object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
    else:
        object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

    return masks, iou_pred, mask_tokens_out, object_score_logits


def _patch_mask_decoder(module: nn.Module) -> int:
    """Patch MaskDecoder.predict_masks to use TRT-friendly reshape ops."""
    patched = 0
    for mod in module.modules():
        if type(mod).__name__ == "MaskDecoder":
            mod.predict_masks = types.MethodType(_predict_masks_patched, mod)
            patched += 1
    return patched


# ---------------------------------------------------------------------------
# Decoder wrapper — fixed API for torch.export
# ---------------------------------------------------------------------------

class _DecoderWrapper(nn.Module):
    """Thin wrapper around SAM2 MaskDecoder for TRT export.

    Bakes in:
    - ``multimask_output=False`` and ``repeat_image=True``
    - The no-mask dense embedding (no PromptEncoder call needed)
    - high_res_features as two separate tensor args instead of a list

    The prompt encoder call is bypassed entirely: when only text_embeds are
    used (points/boxes/masks=None), sparse_embeddings == text_embeds and
    dense_embeddings == no_mask_embed expanded over the batch.
    """

    def __init__(self, decoder: nn.Module, no_mask_embed_weight: torch.Tensor) -> None:
        super().__init__()
        self.decoder = decoder
        # [1, 256, 1, 1] — expanded to [N, 256, 64, 64] at runtime
        self.register_buffer("no_mask_embed", no_mask_embed_weight.reshape(1, -1, 1, 1))

    def forward(
        self,
        image_embeddings: torch.Tensor,  # [1, 256, 64, 64]
        image_pe: torch.Tensor,           # [1, 256, 64, 64]
        sparse_embeddings: torch.Tensor,  # [N, T, 256]
        high_res_s0: torch.Tensor,        # [1, 32, 256, 256]
        high_res_s1: torch.Tensor,        # [1, 64, 128, 128]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = sparse_embeddings.shape[0]
        dense_embeddings = self.no_mask_embed.expand(
            N, -1, image_embeddings.shape[2], image_embeddings.shape[3]
        )
        masks, iou_pred, _, _ = self.decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            repeat_image=True,
            high_res_features=[high_res_s0, high_res_s1],
        )
        # Return single-mask slice (multimask_output=False path)
        return masks[:, 0:1, :, :], iou_pred[:, 0:1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_trt_encoder(image_encoder: torch.nn.Module, dtype: torch.dtype, device: str = "cuda:0") -> torch.nn.Module:
    """Return a TRT-compiled image encoder, loading from cache if available.

    On first call this patches the encoder, exports it via
    ``torch.export.export(..., strict=False)``, compiles with
    ``torch_tensorrt.dynamo.compile()``, and writes an engine file to
    ``_TRT_CACHE_DIR``.  Subsequent calls load from the cache and are instant.

    The ``strict=False`` flag mirrors the official PyTorch-TensorRT SAM2 tutorial
    (pytorch/TensorRT@main examples/dynamo/torch_export_sam2.py).  Without it
    torch.export chokes on custom-op references inside the SAM2 package.

    Args:
        image_encoder: The SAM2 image encoder (HieraDet + FPN neck).
        dtype: The compute dtype (e.g. ``torch.bfloat16``).
        device: CUDA device string (e.g. ``"cuda:0"``).

    Returns:
        TRT-compiled module, or original module if compilation fails.
    """
    try:
        import torch_tensorrt
    except ImportError:
        print("[TRT] torch-tensorrt not installed, falling back to torch.compile")
        return _torch_compile(image_encoder, mode="default", fullgraph=False)

    os.makedirs(_TRT_CACHE_DIR, exist_ok=True)
    dtype_tag = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, "fp32")
    # Embed GPU SM version so engines are never loaded on a different GPU architecture.
    sm = torch.cuda.get_device_capability(torch.device(device))
    trt_path = os.path.join(_TRT_CACHE_DIR, f"sam2_hiera_large_encoder_sm{sm[0]}{sm[1]}_{dtype_tag}.ep")

    if os.path.exists(trt_path):
        print(f"[TRT] Loading cached TRT encoder from {trt_path}")
        try:
            loaded = torch.export.load(trt_path)
            return loaded.module()
        except Exception as e:
            print(f"[TRT] Failed to load cached engine ({e}), recompiling...")

    print("[TRT] Compiling SAM2 image encoder with TensorRT (first run ~5-10 min)...")
    image_encoder = image_encoder.to(device=device, dtype=dtype).eval()

    # Apply source-compatible patches before tracing
    _apply_trt_patches(image_encoder)

    example_input = torch.zeros(1, 3, 1024, 1024, dtype=dtype, device=device)

    try:
        # strict=False is required: SAM2 contains internal dict caches and
        # custom-op references that torch.export cannot trace in strict mode.
        # This is exactly the approach used by the official PyTorch-TensorRT
        # SAM2 tutorial (pytorch/TensorRT examples/dynamo/torch_export_sam2.py).
        with torch.no_grad():
            exported = torch.export.export(
                image_encoder,
                args=(example_input,),
                strict=False,
            )

        trt_encoder = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[
                torch_tensorrt.Input(
                    shape=[1, 3, 1024, 1024],
                    dtype=dtype,
                )
            ],
            enabled_precisions={dtype},
            truncate_double=True,
            device=torch.device(device),
            workspace_size=4 * 1024 ** 3,  # 4 GB
            optimization_level=3,
            # Accumulate matmuls in FP32 to preserve accuracy at BF16/FP16
            use_fp32_acc=True,
        )

        torch_tensorrt.save(trt_encoder, trt_path, inputs=[example_input])
        print(f"[TRT] TRT engine saved to {trt_path}")
        # torch_tensorrt.dynamo.compile() returns a GraphModule directly (already callable).
        # Only torch.export.load() returns an ExportedProgram that needs .module().
        return trt_encoder

    except Exception as e:
        import traceback
        print(f"[TRT] TRT compilation failed: {e}")
        traceback.print_exc()
        print("[TRT] Falling back to torch.compile")
        return _torch_compile(image_encoder, mode="default", fullgraph=False)


class _TRTDecoderAdapter(nn.Module):
    """Drop-in ``nn.Module`` replacement for ``sam_mask_decoder``.

    Accepts the original ``MaskDecoder.forward`` keyword signature and internally
    calls the TRT ``_DecoderWrapper``.

    ``SAM2Base.forward_image()`` calls ``self.sam_mask_decoder.conv_s0 / conv_s1``
    directly to project FPN backbone features.  These convolutions are preserved
    as submodules here so that attribute access still works after the swap.
    """

    def __init__(
        self,
        trt_wrapper: nn.Module,
        conv_s0: nn.Module | None = None,
        conv_s1: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.trt_wrapper = trt_wrapper
        # Keep the projection convolutions that SAM2Base.forward_image() calls directly.
        if conv_s0 is not None:
            self.conv_s0 = conv_s0
        if conv_s1 is not None:
            self.conv_s1 = conv_s1

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,  # ignored — baked into TRT wrapper
        multimask_output: bool = False,          # ignored — baked as False
        repeat_image: bool = True,               # ignored — baked as True
        high_res_features=None,
    ):
        assert high_res_features is not None and len(high_res_features) >= 2, \
            "_TRTDecoderAdapter requires high_res_features (got None)"
        low_res_masks, iou_pred = self.trt_wrapper(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            high_res_features[0],
            high_res_features[1],
        )
        # Return 4-tuple matching the original MaskDecoder.forward() signature.
        # Downstream code in open_world_sam2.py only uses the first two outputs.
        return low_res_masks, iou_pred, None, None


def get_trt_decoder(
    decoder: nn.Module,
    no_mask_embed_weight: torch.Tensor,
    dtype: torch.dtype,
    device: str = "cuda:0",
    max_n_prompts: int = 400,
) -> nn.Module:
    """Return a torch.compile-optimised SAM2 mask decoder wrapper.

    TensorRT compilation of this decoder was investigated but cannot succeed in
    TRT 2.5 because SAM2's TwoWayTransformer cross-attention has a dynamic batch
    dimension (vocab_size × num_tokens) on the query side while the image-feature
    key/value tensors start at batch=1.  TRT fuses the implicit ``expand`` broadcast
    inside SDPA with surrounding SHUFFLE layers into a ``ForeignNode`` for which no
    valid kernel exists.  The official torch-tensorrt SAM2 tutorial (pytorch/TensorRT
    examples/dynamo/torch_export_sam2.py) also compiles only the image encoder.

    ``torch.compile(mode="default")`` achieves good per-image latency (~23 ms vs
    ~52 ms baseline) while keeping recompilation for unseen vocabulary sizes cheap
    (~5–10 ms).  ``max-autotune`` was tried but causes ~200 ms Triton autotuning
    spikes whenever ``--lmm_per_image`` generates a vocabulary size not covered by
    the startup warmup.

    Patches applied to the wrapper (also benefit torch.compile correctness):
    - ``Attention.forward`` — removes the non-traceable ``sdp_kernel`` context manager;
      inlines head separation using last-dim ops.
    - ``TwoWayAttentionBlock.forward`` — explicit ``expand`` before cross-attention so
      the compiler sees clean broadcast ops, not hidden data-dependent ones.
    - ``MaskDecoder.predict_masks`` — replaces ``repeat_interleave`` (dynamic-batch
      reshape) with ``expand``; replaces ``view(b,...)`` with ``unflatten/flatten``.
    - ``LayerNorm2d.forward`` — uses ``F.layer_norm`` to remove weight ``[:, None, None]``
      unsqueeze patterns that interfere with kernel fusion.

    Args:
        decoder: The SAM2 ``MaskDecoder`` module.
        no_mask_embed_weight: ``sam_prompt_encoder.no_mask_embed.weight`` tensor.
        dtype: Compute dtype (e.g. ``torch.bfloat16``).
        device: CUDA device string.
        max_n_prompts: Unused; kept for API compatibility.

    Returns:
        ``torch.compile``-d ``_DecoderWrapper`` ready for inference.
    """
    wrapper = _DecoderWrapper(decoder, no_mask_embed_weight)
    wrapper = wrapper.to(device=device, dtype=dtype).eval()

    # Apply source-compatible patches (improve both correctness and fusion quality)
    n_attn   = _patch_attention(wrapper)
    n_twoway = _patch_two_way_attn_blocks(wrapper)
    n_dec    = _patch_mask_decoder(wrapper)
    n_ln     = _patch_layer_norm_2d(wrapper)
    print(
        f"[TRT] Decoder patches applied: Attention×{n_attn}, "
        f"TwoWayBlock×{n_twoway}, MaskDecoder×{n_dec}, LayerNorm2d×{n_ln}"
    )

    # torch.compile — "default" mode keeps recompilation for unseen vocabulary sizes
    # cheap (~5–10 ms) at the cost of ~6 ms/frame vs max-autotune.  max-autotune
    # causes ~200 ms Triton kernel-search spikes for every new vocab size encountered
    # at runtime (e.g. from --lmm_per_image), which far outweighs the per-frame gain.
    # fullgraph=False avoids FakeTensor issues from SAM2's internal dict caches.
    compiled = _torch_compile(wrapper, mode="default", fullgraph=False)
    print("[TRT] Decoder compiled", flush=True)
    return compiled
