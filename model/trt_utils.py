# Copyright (c) 2025-2026, ETH Zurich (Robotic Systems Lab) & NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Optimise the SAM2 Hiera image encoder (TensorRT) and mask decoder (torch.compile).

Encoder — TensorRT via torch-tensorrt
  Approach follows the official PyTorch-TensorRT SAM2 tutorial:
    https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/torch_export_sam2.py
  Engine cached to ``_TRT_CACHE_DIR``; ~5–10 min first-run, instant on subsequent starts.

Decoder — torch.compile (max-autotune)
  TRT compilation of the SAM2 mask decoder fails in TRT 2.5: the TwoWayTransformer's
  cross-attention has a dynamic batch dim (vocab_size × num_tokens) on the query side
  while image-feature key/value tensors start at batch=1.  TRT fuses the resulting
  implicit ``expand`` broadcast with surrounding SHUFFLE layers into a ``ForeignNode``
  for which no valid kernel exists.  The official torch-tensorrt SAM2 tutorial also
  compiles only the image encoder.  ``torch.compile(mode="max-autotune")`` achieves
  equivalent latency (~23 ms vs ~52 ms baseline) without the 5–10 min build step.

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
        return torch.compile(image_encoder, mode="default", fullgraph=False)

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
        return torch.compile(image_encoder, mode="default", fullgraph=False)


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

    ``torch.compile(mode="max-autotune")`` achieves equivalent or better per-image
    latency (~23 ms vs ~52 ms baseline) and compiles on first use without the 5–10 min
    TRT build step.

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

    # torch.compile — max-autotune for best kernel fusion.
    # fullgraph=False avoids FakeTensor issues from SAM2's internal dict caches.
    compiled = torch.compile(wrapper, mode="max-autotune", fullgraph=False)
    print("[TRT] Decoder compiled with torch.compile(max-autotune)")
    return compiled
