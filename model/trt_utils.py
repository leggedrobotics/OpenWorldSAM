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

    # Blackwell (SM >= 11.0): some ptxas builds reject sm_110a at JIT time.
    # Run a live compile probe instead of a static binary check.
    if torch.cuda.is_available():
        sm = torch.cuda.get_device_capability()
        if sm[0] >= 11 and not _triton_probe_compile():
            return False

    _patch_triton_compiled_kernel_cta_attrs()
    _patch_triton_kernel_metadata_cluster_dims()
    return True


def _patch_triton_compiled_kernel_cta_attrs() -> None:
    """Add __getattr__ to Triton's CompiledKernel so num_ctas/cluster_dims are
    accessible directly on the binary object.

    Root cause: torch/_inductor/runtime/triton_heuristics.py chooses between two
    code paths based on hasattr(binary, "num_ctas"):

      # Branch 1 — safe, has a fallback chain:
      (binary.num_ctas, *get_first_attr(binary, "cluster_dims", "clusterDims"))
      if hasattr(binary, "num_ctas")

      # Branch 2 — crashes on Jetson, no fallback:
      else: (binary.metadata.num_ctas, *binary.metadata.cluster_dims)

    On Jetson, binary.num_ctas does not exist as a direct attribute, forcing
    branch 2.  Triton 3.x on aarch64 (SM87) omits cluster_dims from
    KernelMetadata because thread-block clustering is H100/SM90-only, so
    binary.metadata.cluster_dims raises AttributeError.

    Fix: add __getattr__ to CompiledKernel so that binary.num_ctas,
    binary.cluster_dims and binary.clusterDims are answered via metadata
    (with safe defaults).  hasattr(binary, "num_ctas") then returns True and
    inductor always takes branch 1, never reaching binary.metadata.cluster_dims.

    Patching the KernelMetadata *class* (class attribute or __getattr__ on the
    class) does not work reliably because the binary.metadata instance can come
    from a backend-specific class (e.g. triton.backends.nvidia.compiler) that is
    different from the module-level KernelMetadata we find at import time.
    """
    import sys
    import importlib as _il

    # Force-import known Triton module paths so their classes are in sys.modules
    # before the scan.  This is needed if _triton_available() runs before any
    # Triton kernel has been compiled (which is the typical case at model init).
    for _force_path in (
        "triton.compiler.compiler",
        "triton.runtime.jit",
        "triton.backends.nvidia.compiler",
    ):
        try:
            _il.import_module(_force_path)
        except Exception:
            pass

    _found = False
    for _mod_name in list(sys.modules):
        if "triton" not in _mod_name:
            continue
        _mod = sys.modules.get(_mod_name)
        if _mod is None:
            continue
        _CK = getattr(_mod, "CompiledKernel", None)
        if _CK is None or not isinstance(_CK, type):
            continue
        if getattr(_CK, "_cta_attrs_patched", False):
            _found = True
            continue  # already patched on a prior call

        _orig_ga = vars(_CK).get("__getattr__")  # own __getattr__ only, skip MRO

        def _getattr(self, name: str, _orig=_orig_ga) -> object:
            if name in ("num_ctas", "cluster_dims", "clusterDims"):
                # Use object.__getattribute__ to access metadata without
                # recursing back into this __getattr__.
                try:
                    _meta = object.__getattribute__(self, "metadata")
                except AttributeError:
                    _meta = None
                if name == "num_ctas":
                    return getattr(_meta, "num_ctas", 1)
                # cluster_dims / clusterDims — SM87 has no clusters → (1,1,1)
                return getattr(_meta, "cluster_dims", (1, 1, 1))
            if _orig is not None:
                return _orig(self, name)
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        try:
            _CK.__getattr__ = _getattr
            _CK._cta_attrs_patched = True
            _found = True
        except Exception:
            pass

    if _found:
        print(
            "[compile] CompiledKernel.__getattr__ patched (or already patched) for "
            "num_ctas/cluster_dims — inductor will take Branch 1 on SM87",
            flush=True,
        )
    else:
        print(
            "[compile] WARNING: CompiledKernel not found in loaded Triton modules — "
            "Branch 1 cluster_dims fix not applied",
            flush=True,
        )


def _patch_triton_kernel_metadata_cluster_dims() -> None:
    """Belt-and-suspenders: add ``cluster_dims`` to Triton's backend KernelMetadata.

    ``_patch_triton_compiled_kernel_cta_attrs`` steers inductor to Branch 1
    (``if hasattr(binary, "num_ctas")``) by patching ``CompiledKernel.__getattr__``.
    However, the ``binary`` object at kernel-compile time may be an instance of a
    backend-specific ``CompiledKernel`` subclass that was lazily loaded *after* the
    ``sys.modules`` scan ran, meaning the ``__getattr__`` patch may not cover it.

    This function makes Branch 2 (``binary.metadata.cluster_dims``) also safe by
    adding ``__getattr__`` directly to ``triton.backends.nvidia.compiler.KernelMetadata``
    — the class that ``binary.metadata`` is an instance of on Jetson.  Both branches
    are then safe regardless of which one inductor actually takes at runtime.

    On Jetson Orin (SM87), thread-block clustering is an H100/SM90-only feature,
    so ``cluster_dims`` is always ``(1, 1, 1)``.
    """
    try:
        import triton.backends.nvidia.compiler as _nvc
    except Exception:
        return

    # Class name varies across Triton versions and Jetson builds.
    # Scan all classes in the module for ones that look like kernel metadata:
    # namedtuples have _fields; dataclasses have __annotations__.
    # Characteristic fields: num_ctas, num_warps, shared (always present in metadata).
    _KM = getattr(_nvc, "KernelMetadata", None)
    if _KM is None:
        _meta_fields = {"num_ctas", "num_warps", "shared", "cluster_dims", "num_stages"}
        for _attr_name in dir(_nvc):
            try:
                _cls = getattr(_nvc, _attr_name)
                if not isinstance(_cls, type):
                    continue
                _fields = set(getattr(_cls, "_fields", None) or getattr(_cls, "__annotations__", {}).keys())
                if _fields & _meta_fields:  # at least one characteristic field present
                    _KM = _cls
                    break
            except Exception:
                continue
    if _KM is None:
        print(
            "[compile] triton.backends.nvidia.compiler: no KernelMetadata-like class found; "
            "Branch 2 cluster_dims fallback not applied",
            flush=True,
        )
        return

    if getattr(_KM, "_cluster_dims_patched", False):
        return  # already applied on a prior call

    if hasattr(_KM, "cluster_dims"):
        return  # native field present — nothing to do

    def _km_getattr(self, name: str):
        if name in ("cluster_dims", "clusterDims"):
            # SM87 (Jetson Orin) has no thread-block clustering → always (1,1,1)
            return (1, 1, 1)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    try:
        _KM.__getattr__ = _km_getattr
        _KM._cluster_dims_patched = True
        print(
            "[compile] Patched triton.backends.nvidia.compiler.KernelMetadata.__getattr__ "
            "— cluster_dims=(1,1,1) fallback applied for SM87 (Branch 2 safety)",
            flush=True,
        )
    except Exception as e:
        print(f"[compile] Could not patch KernelMetadata.__getattr__: {e}", flush=True)


import threading as _threading_mod


def _triton_probe_compile() -> bool:
    """Return True if Triton can JIT-compile and run a trivial kernel on the current GPU.

    Blackwell (SM_110 / sm_110a) known issue: some ptxas builds bundled with
    JetPack 7.0 only recognise sm_121 (DGX Spark) and reject sm_110a with:
        ptxas-blackwell fatal: Value 'sm_110a' is not defined for option 'gpu-name'
    A static ptxas --version check cannot detect this; a live kernel compile does.
    The probe runs at model-init time so failures redirect to cudagraphs before
    any segmentation inference attempt.
    """
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _noop(x_ptr, BLOCK: tl.constexpr):
            pid = tl.program_id(axis=0)
            x = tl.load(x_ptr + pid * BLOCK)
            tl.store(x_ptr + pid * BLOCK, x)

        x = torch.ones(1, device="cuda", dtype=torch.float32)
        _noop[(1,)](x, BLOCK=1)
        return True
    except Exception as e:
        print(f"[compile] Triton probe failed ({type(e).__name__}: {e}); "
              "torch.compile will use cudagraphs instead of inductor", flush=True)
        return False


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
    """Wrap a torch.compile(inductor) module; fall back to cudagraphs on the
    first forward pass if inductor compilation fails.

    torch.compile() is lazy: actual kernel compilation happens on the FIRST
    forward call.  Two problems prevent a simple try/except from working:

    1. The error surfaces at inference time (not at the torch.compile() call).
    2. By default, dynamo SWALLOWS compilation errors (printing "backend=
       'inductor' raised:") and silently falls back to eager — so no exception
       ever reaches user code.  We set ``suppress_errors=False`` before
       torch.compile() so that dynamo re-raises instead, letting us catch it.

    Known failure modes on Jetson aarch64 with PyTorch 2.8:
    - TypeError: cannot pickle '_thread.RLock' object  (inductor worker IPC)
    - AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'
      (Triton version mismatch — older Jetson Triton lacks cluster_dims)

    On failure the wrapper transparently re-compiles with the cudagraphs
    backend (which is always compatible) and continues without crashing.
    """

    def __init__(self, compiled: torch.nn.Module, original: torch.nn.Module, fullgraph: bool):
        super().__init__()
        self._compiled = compiled
        self._original = original
        self._fullgraph = fullgraph
        self._failed = False

    def _switch_to_cudagraphs(self, exc: Exception) -> None:
        print(
            f"[compile] inductor failed on first inference "
            f"({type(exc).__name__}: {exc}); falling back to cudagraphs",
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

    def forward(self, *args, **kwargs):
        if not self._failed:
            try:
                return self._compiled(*args, **kwargs)
            except Exception as exc:
                # Catch ANY exception on the first call — inductor failures
                # can surface as TypeError (pickle), AttributeError
                # (KernelMetadata), or other Triton/inductor-version errors.
                self._switch_to_cudagraphs(exc)
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
        # By default dynamo swallows inductor compilation errors and silently
        # falls back to eager (printing "backend='inductor' raised:" but not
        # re-raising).  _InductorWithCudagraphsFallback relies on exceptions
        # propagating to its forward() so it can switch to cudagraphs.
        # suppress_errors=False makes dynamo re-raise instead.
        try:
            torch._dynamo.config.suppress_errors = False
        except Exception:
            pass
        try:
            compiled = torch.compile(module, mode=mode, fullgraph=fullgraph, dynamic=True)
            print("[compile] Jetson+Triton: using torch.compile(inductor, dynamic=True, single-thread)", flush=True)
            # Wrap so that ANY inductor failure on the first forward pass
            # (TypeError, AttributeError, Triton version mismatches, etc.)
            # triggers a transparent recompile with cudagraphs.
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

def _proj_2d(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a Linear-containing module to a 3D tensor via a 2D reshape.

    When ``torch.export`` traces ``nn.Linear`` on a 3D input ``[N, S, C]`` with
    dynamic batch N it generates::

        weight.unsqueeze(0).expand(N, ...) followed by bmm

    TRT fuses this weight-broadcast with any subsequent SHUFFLE ops (``unflatten``,
    ``transpose``) into a ``ForeignNode`` for which no valid kernel exists when N is
    a symbolic dynamic dimension.

    Reshaping to ``[N*S, C]`` first uses the native 2D ``IMatrixMultiplyLayer``
    path — the weight is never broadcast — then the output is reshaped back.

    Supports arbitrary batch prefixes (e.g. ``[N, S, C]`` or ``[N, C]``).
    """
    shape = x.shape          # (..., S, C)
    # All callers pass contiguous tensors (image keys guaranteed by the
    # TwoWayTransformer patch; q/v/queries/out_proj input are always contiguous).
    # Use plain view so the exported graph contains aten.view (a pure descriptor
    # SHUFFLE with no data movement) rather than aten._reshape_copy (a copy+reshape
    # SHUFFLE that TRT fuses with the weight's permute SHUFFLE into a ForeignNode
    # with no valid kernel).
    x_2d  = x.view(-1, shape[-1])   # [N*S, C]
    out   = module(x_2d)             # [N*S, C']
    return out.view(*shape[:-1], out.shape[-1])  # [..., S, C']


def _attention_forward_patched(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention.forward without the non-traceable sdp_kernel context manager.

    Inlines _separate_heads / _recombine_heads using unflatten+transpose / transpose+flatten
    instead of shape-extraction + reshape.  The original pattern
        b, n, c = x.shape; x.reshape(b, n, num_heads, c // num_heads)
    causes TRT to fuse the ops into a ForeignNode with no valid kernel when the
    batch dim (b = vocab_size) is a dynamic symbolic dimension.  Using only
    last-dim operations avoids this.

    All linear projections (q/k/v/out) are applied via ``_proj_2d`` which avoids
    a second ForeignNode: ``nn.Linear`` on a 3D input with dynamic batch N traces
    as weight.expand(N,...)+bmm which TRT fuses with subsequent SHUFFLE ops
    (unflatten, transpose) into an uncompilable node.  Projecting via a 2D reshape
    uses the native IMatrixMultiplyLayer path instead.
    """
    q = _proj_2d(self.q_proj, q)
    k = _proj_2d(self.k_proj, k)
    v = _proj_2d(self.v_proj, v)
    head_dim = q.shape[-1] // self.num_heads
    # separate heads: [B, N, C] -> [B, heads, N, head_dim]
    q = q.unflatten(-1, [self.num_heads, head_dim]).transpose(1, 2)
    k = k.unflatten(-1, [self.num_heads, head_dim]).transpose(1, 2)
    v = v.unflatten(-1, [self.num_heads, head_dim]).transpose(1, 2)
    dropout_p = self.dropout_p if self.training else 0.0
    # Drop the non-traceable context manager; PyTorch will auto-select the best kernel.
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    # recombine heads: [B, heads, N, head_dim] -> [B, N, C]
    # .contiguous() after transpose ensures flatten traces as aten.view (pure
    # descriptor SHUFFLE) rather than aten._reshape_copy (copy SHUFFLE that TRT
    # fuses with adjacent ops into an uncompilable ForeignNode).
    out = out.transpose(1, 2).contiguous().flatten(-2)
    out = _proj_2d(self.out_proj, out)
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
    # Expand image features to N to avoid implicit SDPA broadcast → TRT ForeignNode.
    # .contiguous() forces the expanded view into a dense layout so _proj_2d's
    # view() traces as aten.view (descriptor SHUFFLE) rather than aten._reshape_copy
    # (copy SHUFFLE), which TRT fuses with surrounding ops into an uncompilable
    # ForeignNode when the batch dim N is dynamic.
    keys_n = keys.expand(N, -1, -1).contiguous()
    key_pe_n = key_pe.expand(N, -1, -1).contiguous()
    q = queries + query_pe
    k = keys_n + key_pe_n
    attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys_n)
    queries = queries + attn_out
    queries = self.norm2(queries)

    # MLP block — use _proj_2d to avoid weight-broadcast ForeignNode when queries
    # is [N, T_tokens, C] with dynamic N (same issue as linear projections in Attention).
    mlp_out = _proj_2d(self.mlp, queries)
    queries = queries + mlp_out
    queries = self.norm3(queries)

    # Cross attention block, image embedding attending to tokens
    # keys_n is already [N,4096,256] (expanded at the top of this function).
    # Use keys_n explicitly rather than keys to avoid any stale [1,...] reference
    # if this block is called with keys still at batch=1 in a different code path.
    k = keys_n + key_pe_n
    attn_out = self.cross_attn_image_to_token(q=k, k=queries + query_pe, v=queries)
    keys = keys_n + attn_out
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


def _two_way_transformer_forward_patched(
    self,
    image_embedding: torch.Tensor,
    image_pe: torch.Tensor,
    point_embedding: torch.Tensor,
):
    """TwoWayTransformer.forward with contiguous image keys.

    The original ``flatten(2).permute(0, 2, 1)`` leaves ``image_embedding`` with
    strides ``(0, 1, 4096)`` — non-contiguous.  When this tensor propagates into
    ``_proj_2d`` the exported aten graph contains ``aten._reshape_copy`` which TRT
    fuses with the weight's permute SHUFFLE into a ForeignNode that has no valid
    kernel::

        ForeignNode[k_proj/..[SHUFFLE(permute)] + [SHUFFLE(_reshape_copy)]]

    Adding ``.contiguous()`` after each ``permute`` makes the keys and image PE
    ``[N, 4096, 256]``-contiguous, so ``_proj_2d`` can use plain ``aten.view`` (a
    pure descriptor SHUFFLE with no data movement).  TRT then sees the standard
    ``view → IMatrixMultiplyLayer(weight, TRANSPOSE)`` pattern it can compile.
    """
    # One-time diagnostic: confirm this patched path is being traced (not the
    # original TwoWayTransformer.forward).  Printed only during torch.export,
    # not at every inference call, because the module is replaced by the TRT engine.
    if not getattr(self, "_trt_patch_logged", False):
        print("[TRT] TwoWayTransformer forward patch active (contiguous image keys)")
        object.__setattr__(self, "_trt_patch_logged", True)
    bs, c, h, w = image_embedding.shape
    image_embedding = image_embedding.flatten(2).permute(0, 2, 1).contiguous()
    image_pe        = image_pe.flatten(2).permute(0, 2, 1).contiguous()

    queries = point_embedding
    keys    = image_embedding

    for layer in self.layers:
        queries, keys = layer(
            queries=queries,
            keys=keys,
            query_pe=point_embedding,
            key_pe=image_pe,
        )

    q = queries + point_embedding
    k = keys + image_pe
    attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
    queries  = queries + attn_out
    queries  = self.norm_final_attn(queries)
    return queries, keys


def _patch_two_way_transformer(module: nn.Module) -> int:
    """Patch TwoWayTransformer.forward to produce contiguous image-key tensors.

    Uses a name-based check (``type(mod).__name__``) rather than ``isinstance``
    to avoid false negatives when the same class is imported via two different
    Python paths (e.g. ``model.segment_anything_2.sam2.*`` vs ``sam2.*``).
    """
    patched = 0
    for mod in module.modules():
        if type(mod).__name__ == "TwoWayTransformer":
            mod.forward = types.MethodType(_two_way_transformer_forward_patched, mod)
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
    # .contiguous() materialises the expand into a dense tensor so TRT sees a plain
    # IExpandLayer with no stride=0 dims — breaking the ForeignNode that forms when
    # the expand feeds through the transformer into the DECONV shape-tensor path.
    output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1).contiguous()
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    N = sparse_prompt_embeddings.shape[0]
    # image_embeddings and image_pe are already [N, C, H, W] (pre-expanded by
    # _DecoderWrapper.forward) — no expand needed here.
    src = image_embeddings + dense_prompt_embeddings
    pos_src = image_pe
    _, c, h, w = image_embeddings.shape

    hs, src = self.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, s, :]
    mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

    # unflatten(-1, [h, w]) ONLY splits the last dimension of [N, C, H*W].
    # Critically it never references N as an argument, so TRT's shape-tensor path
    # never traces back through N's producer chain (expand → ForeignNode).
    # The earlier view(N, 256, h, w) passed N explicitly, causing TRT to resolve
    # it as a shape tensor → hit the expand chain → ForeignNode → DECONV type fail.
    src = src.transpose(1, 2).contiguous().unflatten(-1, [h, w])
    if not self.use_high_res_features:
        upscaled_embedding = self.output_upscaling(src)
    else:
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        # SM_110 (Blackwell) TRT cannot compile BF16 ConvTranspose2d: all tactics
        # fail with type.cpp:186 infer_type.  Fix: cast to FP32 for the two DECONV
        # calls and cast back.  SM_87 (Orin) supports BF16 DECONV natively so the
        # cast is skipped there to avoid the unnecessary type-conversion overhead.
        # The capability check is evaluated at torch.export / torch.compile trace
        # time so the correct branch is baked into the compiled graph.
        _sm_major = torch.cuda.get_device_capability(src.device)[0]
        _need_fp32_deconv = (_sm_major >= 11) and (src.dtype == torch.bfloat16)
        if _need_fp32_deconv:
            _dtype = src.dtype
            _dc1_bias = dc1.bias.float() if dc1.bias is not None else None
            _dc2_bias = dc2.bias.float() if dc2.bias is not None else None
            _deconv1 = F.conv_transpose2d(
                src.float(), dc1.weight.float(), _dc1_bias,
                dc1.stride, dc1.padding, dc1.output_padding, dc1.groups, dc1.dilation,
            ).to(_dtype)
            upscaled_embedding = act1(ln1(_deconv1 + feat_s1))
            _deconv2 = F.conv_transpose2d(
                upscaled_embedding.float(), dc2.weight.float(), _dc2_bias,
                dc2.stride, dc2.padding, dc2.output_padding, dc2.groups, dc2.dilation,
            ).to(_dtype)
        else:
            _deconv1 = dc1(src)
            upscaled_embedding = act1(ln1(_deconv1 + feat_s1))
            _deconv2 = dc2(upscaled_embedding)
        upscaled_embedding = act2(_deconv2 + feat_s0)

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
        # image_embeddings / image_pe must be [N, ...]: the TwoWayTransformer updates
        # them per-prompt, and pre-expanding here keeps predict_masks free of the
        # ISliceLayer-sourced shape tensors that otherwise create a ForeignNode in the
        # DECONV path on SM_110.
        image_emb_n  = image_embeddings.expand(N, -1, -1, -1).contiguous()
        image_pe_n   = image_pe.expand(N, -1, -1, -1).contiguous()
        # The high-res FPN features (feat_s0 ~500MB, feat_s1 ~250MB at N=120) are only
        # consumed by the final DECONV skip-add (`_deconv + feat`), which broadcasts
        # [1,C,H,W] over the N masks natively.  The decoder is bandwidth-bound, so by
        # default we leave them at [1,...] and let the add broadcast instead of
        # materialising the largest tensors in the graph.  Set OWSAM_TRT_EXPAND_HIRES=1
        # to fall back to the fully-expanded path if a TRT build rejects the broadcast.
        if os.environ.get("OWSAM_TRT_EXPAND_HIRES", "0") == "1":
            feat_s0 = high_res_s0.expand(N, -1, -1, -1).contiguous()
            feat_s1 = high_res_s1.expand(N, -1, -1, -1).contiguous()
        else:
            feat_s0, feat_s1 = high_res_s0, high_res_s1
        dense_embeddings = self.no_mask_embed.expand(
            N, -1, image_embeddings.shape[2], image_embeddings.shape[3]
        ).contiguous()
        masks, iou_pred, _, _ = self.decoder.predict_masks(
            image_embeddings=image_emb_n,
            image_pe=image_pe_n,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            repeat_image=True,
            high_res_features=[feat_s0, feat_s1],
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

        # Pre-strip aten assert nodes before torch_tensorrt.dynamo.compile().
        # PyTorch 2.10 + torch_tensorrt 2.10 bug on Jetson SBSA: the internal
        # remove_assert_nodes pass calls erase_node() on _assert_scalar /
        # _assert_tensor_metadata nodes.  erase_node() calls _update_args_kwargs
        # at C level which hits KeyError on a stale 'select_3' node reference,
        # causing a SystemError that aborts compilation entirely.
        # Removing these nodes here means remove_assert_nodes finds nothing to
        # erase, bypassing the bug without affecting model correctness.
        _assert_targets = {
            torch.ops.aten._assert_scalar.default,
            torch.ops.aten._assert_tensor_metadata.default,
        }
        _nodes_to_strip = [
            _n for _n in exported.graph_module.graph.nodes
            if _n.target in _assert_targets
        ]
        if _nodes_to_strip:
            for _n in list(_nodes_to_strip):
                # Manually unlink from input nodes' users maps, then clear args,
                # so _update_args_kwargs at C level has no stale references.
                for _inp in list(_n.all_input_nodes):
                    _inp.users.pop(_n, None)
                object.__setattr__(_n, "_args", ())
                object.__setattr__(_n, "_kwargs", {})
                try:
                    exported.graph_module.graph.erase_node(_n)
                except Exception:
                    pass
            try:
                exported.graph_module.graph.eliminate_dead_code()
                exported.graph_module.graph.lint()
                exported.graph_module.recompile()
            except Exception:
                pass
            print(f"[TRT] Pre-stripped {len(_nodes_to_strip)} assert nodes from encoder graph")

        # Jetson Orin uses a unified (CPU+GPU shared) memory architecture.
        # 4 GB workspace (appropriate for discrete desktop GPUs) forces TRT to
        # make memory-layout choices optimised for high-bandwidth GDDR — wrong
        # for the iGPU.  Use a smaller workspace on Jetson so TRT optimises for
        # the actual available GPU memory bandwidth.
        _on_jetson = os.path.isfile("/etc/nv_tegra_release")
        if _on_jetson:
            _sm = torch.cuda.get_device_capability(torch.device(device))
            # Thor (SM_110, 64 GB unified memory) → 2 GB; Orin (SM_87, 32 GB) → 512 MB
            _workspace = (2 * 1024 ** 3) if _sm[0] >= 11 else (512 * 1024 ** 2)
        else:
            _workspace = 4 * 1024 ** 3

        # torch_tensorrt 2.10+: use_explicit_typing defaults to True, which encodes
        # per-op dtypes from the exported graph directly.  Passing enabled_precisions
        # with a non-FP32/FP4 dtype (e.g. bfloat16) then raises AssertionError.
        # For BF16/FP16 the precision is already in the network definition, so omit
        # enabled_precisions.  Only pass it for FP32 to constrain TRT to full precision.
        _enc_precision = {} if dtype != torch.float32 else {"enabled_precisions": {torch.float32}}

        trt_encoder = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[
                torch_tensorrt.Input(
                    shape=[1, 3, 1024, 1024],
                    dtype=dtype,
                )
            ],
            **_enc_precision,
            truncate_double=True,
            device=torch.device(device),
            workspace_size=_workspace,
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


class _StaticTRTDecoder(nn.Module):
    """TRT decoder compiled with a fixed N, with torch.compile fallback.

    When the runtime sparse_embeddings batch dim matches ``static_n`` the fast
    TRT engine is used.  Any other N (e.g. after a vocab-size change) falls back
    to the torch.compile path so inference never hard-errors.

    Controlled by ``OWSAM_NUM_CLASSES`` × ``OWSAM_NUM_TOKENS`` (default 5 × 20 = 100).
    """

    def __init__(
        self,
        trt_module: nn.Module,
        fallback_module: nn.Module,
        static_n: int,
    ) -> None:
        super().__init__()
        self.trt_module = trt_module
        self.fallback_module = fallback_module
        self.static_n = static_n

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_embeddings: torch.Tensor,
        high_res_s0: torch.Tensor,
        high_res_s1: torch.Tensor,
    ) -> tuple:
        if sparse_embeddings.shape[0] == self.static_n:
            return self.trt_module(
                image_embeddings, image_pe, sparse_embeddings, high_res_s0, high_res_s1
            )
        return self.fallback_module(
            image_embeddings, image_pe, sparse_embeddings, high_res_s0, high_res_s1
        )


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
    """Return a TRT-compiled (Jetson) or torch.compile-optimised SAM2 mask decoder.

    Desktop (TRT 2.5): TRT compilation fails because SAM2's TwoWayTransformer
    cross-attention has a dynamic batch dim (vocab_size × num_tokens) on the query
    side while image features start at batch=1.  TRT fuses the implicit ``expand``
    broadcast inside SDPA with SHUFFLE layers into a ``ForeignNode`` with no valid
    kernel.  ``torch.compile(mode="default")`` is used instead (~23 ms vs ~52 ms
    baseline on RTX 6000).

    Jetson (TRT 10.x): The patches applied below (explicit ``expand`` in
    ``TwoWayAttentionBlock``, ``unflatten/flatten`` in ``predict_masks``) expose the
    broadcast and reshape ops in TRT-native form.  TRT 10.x may succeed where 2.5
    could not.  We attempt TRT compilation first and fall back to ``torch.compile``
    if it fails.  The engine is cached to ``_TRT_CACHE_DIR`` so subsequent starts
    are instant.

    Patches applied (benefit both TRT tracing and torch.compile fusion):
    - ``Attention.forward`` — removes non-traceable ``sdp_kernel`` context manager;
      inlines head-separation using last-dim ops only.
    - ``TwoWayAttentionBlock.forward`` — explicit ``keys.expand(N, ...)`` before
      cross-attention so TRT sees a plain broadcast, not a hidden data-dependent one.
    - ``MaskDecoder.predict_masks`` — replaces ``repeat_interleave`` with ``expand``;
      replaces ``view(b, ...)`` with ``unflatten/flatten`` (no dynamic-b capture).
    - ``LayerNorm2d.forward`` — ``F.layer_norm`` removes unsqueeze patterns that
      interfere with kernel fusion.

    Args:
        decoder: The SAM2 ``MaskDecoder`` module.
        no_mask_embed_weight: ``sam_prompt_encoder.no_mask_embed.weight`` tensor.
        dtype: Compute dtype (e.g. ``torch.bfloat16``).
        device: CUDA device string.
        max_n_prompts: Upper bound on N = vocab_size × num_tokens used for the TRT
            dynamic-shape profile (default 400 = 20 classes × 20 tokens).

    Returns:
        TRT-compiled or ``torch.compile``-d ``_DecoderWrapper`` ready for inference.
    """
    wrapper = _DecoderWrapper(decoder, no_mask_embed_weight)
    wrapper = wrapper.to(device=device, dtype=dtype).eval()

    # Apply source-compatible patches (improve both correctness and fusion quality)
    n_attn        = _patch_attention(wrapper)
    n_twoway      = _patch_two_way_attn_blocks(wrapper)
    n_dec         = _patch_mask_decoder(wrapper)
    n_ln          = _patch_layer_norm_2d(wrapper)
    n_transformer = _patch_two_way_transformer(wrapper)
    print(
        f"[TRT] Decoder patches applied: Attention×{n_attn}, "
        f"TwoWayBlock×{n_twoway}, MaskDecoder×{n_dec}, "
        f"LayerNorm2d×{n_ln}, TwoWayTransformer×{n_transformer}"
    )

    # On Jetson, optionally attempt static-N TRT compilation.
    # The static-N TRT engine was originally compiled assuming the sparse-embedding
    # sequence length is T=1 (matching a `fixed` cross-attention path in
    # open_world_sam2.py).  The trained OWSAM weights actually rely on a T=N skip
    # connection broadcast inside the cross-attention block, so the model's real
    # runtime input is `[N, N, D]`, not `[N, 1, D]`.  Running the T=1 engine with
    # T=N input crashes with a shape mismatch, and rebuilding the engine for
    # T=N inflates the transformer sequence length from 6 to (6 + N) with no
    # correctness benefit.  Opt-in via OWSAM_USE_TRT_DECODER=1 only if you have
    # separately rebuilt the cache for the current model shape.
    _on_jetson = os.path.isfile("/etc/nv_tegra_release")
    if _on_jetson and os.environ.get("OWSAM_USE_TRT_DECODER", "0") == "1":
        try:
            _n_cls = int(os.environ.get("OWSAM_NUM_CLASSES", "5"))
            _n_tok = int(os.environ.get("OWSAM_NUM_TOKENS", "20"))
        except ValueError:
            _n_cls, _n_tok = 5, 20
        n_static = _n_cls * _n_tok
        _trt = _try_trt_decoder(wrapper, dtype, device, max_n_prompts, n_static=n_static)
        if _trt is not None:
            print(f"[TRT] Decoder compiled (TensorRT static N={n_static})", flush=True)
            # Build a torch.compile fallback for vocab sizes other than n_static
            _fallback = _torch_compile(wrapper, mode="default", fullgraph=False)
            return _StaticTRTDecoder(_trt, _fallback, n_static)
        print("[TRT] TRT decoder failed; falling back to torch.compile", flush=True)

    # Fallback: torch.compile with inductor/cudagraphs backend.
    # "default" mode keeps recompilation for new vocab sizes cheap (~5–10 ms).
    # fullgraph=False avoids FakeTensor issues from SAM2's internal dict caches.
    compiled = _torch_compile(wrapper, mode="default", fullgraph=False)
    print("[TRT] Decoder compiled (torch.compile)", flush=True)
    return compiled


def _try_trt_decoder(
    wrapper: nn.Module,
    dtype: torch.dtype,
    device: str,
    max_n_prompts: int,
    n_static: "int | None" = None,
) -> "nn.Module | None":
    """Attempt TRT compilation of the patched ``_DecoderWrapper``.

    Returns the compiled module on success, or ``None`` if TRT compilation fails
    (triggering the torch.compile fallback in ``get_trt_decoder``).

    Shape assumptions (verified against open_world_sam2.py):
      - image_embeddings:  [1, 256, 64, 64]   — static (SAM2 backbone output)
      - image_pe:          [1, 256, 64, 64]   — static (dense positional encoding)
      - sparse_embeddings: [N, 1, 256]        — N = vocab_size × num_tokens
      - high_res_s0:       [1, 32, 256, 256]  — static (FPN level 0)
      - high_res_s1:       [1, 64, 128, 128]  — static (FPN level 1)

    When ``n_static`` is provided the decoder is exported with fully static shapes
    (no dynamic dims).  Static shapes eliminate all ForeignNode errors: the expand
    ops inside the TwoWayTransformer that taint N's symbolic path become trivially
    constant, allowing TRT to infer all output shapes at compile time.

    When ``n_static`` is None a dynamic N profile is used (legacy; kept for
    reference but TRT 10.x still produces ForeignNodes for this path).
    """
    try:
        import torch_tensorrt
    except ImportError:
        return None

    try:
        _num_tokens = int(os.environ.get("OWSAM_NUM_TOKENS", "20"))
    except ValueError:
        _num_tokens = 20

    # FP32 matmul accumulation: accurate (default) but slower.  The decoder output
    # is mask logits that get sigmoid+softmax'd downstream, so BF16 accumulation is
    # usually quality-neutral and meaningfully faster — opt in via OWSAM_TRT_FP32_ACC=0.
    _fp32_acc = os.environ.get("OWSAM_TRT_FP32_ACC", "1") == "1"

    os.makedirs(_TRT_CACHE_DIR, exist_ok=True)
    dtype_tag = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, "fp32")
    sm = torch.cuda.get_device_capability(torch.device(device))
    # Cache key encodes N: static engines use "nNNN", dynamic use "dyn".
    n_tag = f"n{n_static}" if n_static is not None else f"dyn_tok{_num_tokens}"
    acc_tag = "accfp32" if _fp32_acc else "accbf16"
    # The high-res-feature broadcast vs expand choice changes the graph, so it must
    # be part of the cache key (avoid loading a stale engine built the other way).
    hires_tag = "exh" if os.environ.get("OWSAM_TRT_EXPAND_HIRES", "0") == "1" else "bch"
    trt_path = os.path.join(
        _TRT_CACHE_DIR,
        f"sam2_decoder_sm{sm[0]}{sm[1]}_{dtype_tag}_{n_tag}_{acc_tag}_{hires_tag}.ep",
    )

    if os.path.exists(trt_path):
        print(f"[TRT] Loading cached TRT decoder from {trt_path}")
        try:
            loaded = torch.export.load(trt_path)
            print("[TRT] TRT decoder loaded from cache")
            return loaded.module()
        except Exception as e:
            print(f"[TRT] Failed to load cached decoder ({e}); recompiling...")

    # Sequence length T of sparse_prompt_embeddings [N, T, 256].
    # With the restored T=N cross-attention broadcast in open_world_sam2.py
    # (`[N,1,D] + [N,D] -> [N,N,D]`), the SAM2 prompt encoder emits T == N, not 1.
    # The static engine must be exported for that runtime shape, otherwise it
    # mismatches at inference.  (Legacy dynamic path keeps T=1.)
    T = n_static if n_static is not None else 1
    _N_export = n_static if n_static is not None else min(5 * _num_tokens, max_n_prompts)

    if n_static is not None:
        print(f"[TRT] Compiling SAM2 decoder with TensorRT (static N={n_static}, ~2-5 min)...")
    else:
        print("[TRT] Compiling SAM2 decoder with TensorRT (dynamic N, ~2-5 min)...")

    # Jetson unified-memory workspace; _try_trt_decoder is only called from
    # get_trt_decoder() when _on_jetson is True, but mirror encoder structure for consistency.
    _on_jetson_dec = os.path.isfile("/etc/nv_tegra_release")
    if _on_jetson_dec:
        _sm = torch.cuda.get_device_capability(torch.device(device))
        # Thor (SM_110, 122 GB): 8 GB — decoder needs >2 GB for some tactics;
        # Orin (SM_87, 32 GB): 512 MB
        _workspace = (8 * 1024 ** 3) if _sm[0] >= 11 else (512 * 1024 ** 2)
    else:
        _workspace = 4 * 1024 ** 3

    _dev = torch.device(device)
    eg_image_emb   = torch.zeros(1,         256,  64,  64, dtype=dtype, device=_dev)
    eg_image_pe    = torch.zeros(1,         256,  64,  64, dtype=dtype, device=_dev)
    eg_sparse_emb  = torch.zeros(_N_export, T,   256,      dtype=dtype, device=_dev)
    eg_high_res_s0 = torch.zeros(1,         32,  256, 256, dtype=dtype, device=_dev)
    eg_high_res_s1 = torch.zeros(1,         64,  128, 128, dtype=dtype, device=_dev)

    try:
        with torch.no_grad():
            if n_static is not None:
                # Static export: no dynamic_shapes — all dims are constants at compile time.
                # This eliminates the ForeignNode caused by dynamic N propagating through
                # the cross-attention expand + slice chain inside TwoWayTransformer.
                exported = torch.export.export(
                    wrapper,
                    args=(eg_image_emb, eg_image_pe, eg_sparse_emb,
                          eg_high_res_s0, eg_high_res_s1),
                    strict=False,
                )
            else:
                # Dynamic export: N is a symbolic dimension (legacy path).
                from torch.export import Dim
                _N_dim = Dim("N", min=_num_tokens, max=max_n_prompts)
                exported = torch.export.export(
                    wrapper,
                    args=(eg_image_emb, eg_image_pe, eg_sparse_emb,
                          eg_high_res_s0, eg_high_res_s1),
                    dynamic_shapes={
                        "image_embeddings":  {},
                        "image_pe":          {},
                        "sparse_embeddings": {0: _N_dim},
                        "high_res_s0":       {},
                        "high_res_s1":       {},
                    },
                    strict=False,
                )

        # Same assert-node pre-strip as in get_trt_encoder (see comment there).
        _assert_targets_dec = {
            torch.ops.aten._assert_scalar.default,
            torch.ops.aten._assert_tensor_metadata.default,
        }
        _nodes_to_strip_dec = [
            _n for _n in exported.graph_module.graph.nodes
            if _n.target in _assert_targets_dec
        ]
        if _nodes_to_strip_dec:
            for _n in list(_nodes_to_strip_dec):
                for _inp in list(_n.all_input_nodes):
                    _inp.users.pop(_n, None)
                object.__setattr__(_n, "_args", ())
                object.__setattr__(_n, "_kwargs", {})
                try:
                    exported.graph_module.graph.erase_node(_n)
                except Exception:
                    pass
            try:
                exported.graph_module.graph.eliminate_dead_code()
                exported.graph_module.graph.lint()
                exported.graph_module.recompile()
            except Exception:
                pass
            print(f"[TRT] Pre-stripped {len(_nodes_to_strip_dec)} assert nodes from decoder graph")

        # Same enabled_precisions restriction as encoder (see comment there).
        _dec_precision = {} if dtype != torch.float32 else {"enabled_precisions": {torch.float32}}

        if n_static is not None:
            _sparse_input = torch_tensorrt.Input(shape=[n_static, T, 256], dtype=dtype)
        else:
            _N_min = _num_tokens
            _N_opt = min(5 * _num_tokens, max_n_prompts)
            _sparse_input = torch_tensorrt.Input(
                min_shape=[_N_min, T, 256],
                opt_shape=[_N_opt, T, 256],
                max_shape=[max_n_prompts, T, 256],
                dtype=dtype,
            )

        trt_decoder = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[
                torch_tensorrt.Input(shape=[1, 256, 64, 64], dtype=dtype),
                torch_tensorrt.Input(shape=[1, 256, 64, 64], dtype=dtype),
                _sparse_input,
                torch_tensorrt.Input(shape=[1, 32, 256, 256], dtype=dtype),
                torch_tensorrt.Input(shape=[1, 64, 128, 128], dtype=dtype),
            ],
            **_dec_precision,
            truncate_double=True,
            device=_dev,
            workspace_size=_workspace,
            optimization_level=3,
            use_fp32_acc=_fp32_acc,
        )

        torch_tensorrt.save(
            trt_decoder, trt_path,
            inputs=[eg_image_emb, eg_image_pe, eg_sparse_emb,
                    eg_high_res_s0, eg_high_res_s1],
        )
        print(f"[TRT] TRT decoder engine saved to {trt_path}")
        return trt_decoder

    except Exception as e:
        import traceback
        print(f"[TRT] Decoder TRT compilation failed: {e}")
        traceback.print_exc()
        return None
