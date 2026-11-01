# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
    APPLY_TOP_P: tl.constexpr = False,
    top_p_ptr=None,
    APPLY_TAU: tl.constexpr = False,
    tau_ptr=None,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        if APPLY_TAU:
            # dflash2-eval (DFL2_LIVE_FILE "tau"): draft distribution softmax(scores / (tau * T)); the scaled scores
            # are also what the rejection sampler gets, so outputs stay exact.
            scores = scores / tl.load(tau_ptr)
        if APPLY_TOP_P:
            # dflash2-eval (DFL2_DRAFT_TOPP): draft truncation. Keep the smallest set of candidates whose mass
            # under this step's distribution (request temperature) reaches top_p, like the target's own top-p.
            # The truncated scores are also what the cache kernel hands the rejection sampler, so outputs stay
            # exact; drafts the target's nucleus would reject are no longer proposed.
            top_p = tl.load(top_p_ptr)
            if temperature != 0.0:
                live = mask & valid & (scores > float("-inf"))
                z = scores / temperature
                zmax = tl.max(tl.where(live, z, float("-inf")), axis=0)
                e = tl.where(live, tl.exp(z - zmax), 0.0)
                tot = tl.sum(e, axis=0)
                pr = tl.where(tot > 0, e / tot, 0.0)
                ps = tl.sort(pr, descending=True)
                cs = tl.cumsum(ps, axis=0)
                kept = (cs - ps) < top_p
                thr = tl.min(tl.where(kept, ps, 2.0), axis=0)
                scores = tl.where(live & (pr >= thr), scores, float("-inf"))
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sample_pos = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            sample_pos,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class CandidateSampler:
    """The shared DFlash2/LiLiCorr candidate walk and realized proposal cache."""

    def __init__(
        self, max_num_reqs: int, num_steps: int, top_k: int, device: torch.device
    ):
        self.num_steps = num_steps
        self.top_k = top_k
        self.scores = torch.empty(
            max_num_reqs, num_steps, top_k, dtype=torch.float32, device=device
        )
        self.cached_candidate_ids = torch.zeros(
            self.scores.shape, dtype=torch.int64, device=device
        )
        import os as _os

        # dflash2-eval: draft truncation. DFL2_DRAFT_TOPP = fixed top-p; DFL2_DRAFT_TOPP_FILE = JSON {"top_p": x}
        # re-read every 0.5 s into the device buffer the captured walk kernel reads (live switching; 1.0 = off).
        self.top_p_file = _os.environ.get("DFL2_LIVE_FILE", "") or _os.environ.get("DFL2_DRAFT_TOPP_FILE", "")
        top_p = float(_os.environ.get("DFL2_DRAFT_TOPP", "1.0"))
        self.apply_top_p = top_p < 1.0 or bool(self.top_p_file)
        self.top_p_buf = torch.full((1,), top_p, dtype=torch.float32, device=device)
        self.apply_tau = bool(self.top_p_file)
        self.tau_buf = torch.ones((1,), dtype=torch.float32, device=device)
        if self.top_p_file:
            import json as _json
            import threading as _threading
            import time as _time

            def _watch(buf=self.top_p_buf, path=self.top_p_file, dev=device, tau_buf=self.tau_buf):
                torch.cuda.set_device(dev)
                side = torch.cuda.Stream(dev)  # own stream: no implicit sync with the serving streams
                last = None
                while True:
                    try:
                        st = _os.stat(path).st_mtime_ns
                        if st != last:
                            last = st
                            cfg = _json.load(open(path))
                            v = float(cfg.get("top_p", 1.0))
                            tau = float(cfg.get("tau", 1.0))
                            with torch.cuda.stream(side):
                                buf.fill_(v)
                                tau_buf.fill_(tau)
                            side.synchronize()
                    except Exception:  # noqa: BLE001 (missing or half-written file: keep the last value)
                        pass
                    _time.sleep(0.5)

            _threading.Thread(target=_watch, daemon=True, name="dfl2-topp").start()

    def sample(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
        sample_pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor | None,
        use_fp64: bool,
    ) -> None:
        candidate_ids = candidate_ids.contiguous()
        sample_pos = sample_pos.contiguous()
        idx_mapping = idx_mapping.contiguous()
        block_k = triton.next_power_of_2(self.top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            sample_pos,
            idx_mapping,
            temperature,
            seeds,
            draft_tokens,
            self.scores,
            num_steps=self.num_steps,
            top_k=self.top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=draft_logits is not None,
            USE_FP64=use_fp64,
            APPLY_TOP_P=self.apply_top_p,
            top_p_ptr=self.top_p_buf,
            APPLY_TAU=self.apply_tau,
            tau_ptr=self.tau_buf,
            num_warps=1,
        )

        if draft_logits is not None:
            self._cache_draft_logits(
                candidate_ids, num_reqs * self.num_steps, idx_mapping, draft_logits
            )

    def _cache_draft_logits(
        self,
        candidate_ids: torch.Tensor,
        num_sample: int,
        idx_mapping: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> None:
        block_k = triton.next_power_of_2(self.top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self.cached_candidate_ids,
            candidate_ids,
            self.scores,
            idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_steps,
            top_k=self.top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"
    _candidate_top_k_key = "selector_top_k"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        import os as _os

        # dflash2-eval: DFL2_TOPK overrides the selector's candidate count (checkpoint: 16)
        self.top_k = int(_os.environ.get("DFL2_TOPK") or draft_config[self._candidate_top_k_key])
        self.candidate_sampler = CandidateSampler(
            self.max_num_reqs, self.num_speculative_steps, self.top_k, device
        )

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        model.model.candidate_selector.top_k = self.top_k  # dflash2-eval: DFL2_TOPK
        import dfl2_opt  # dflash2-eval: opt-in draft-side speedups (DFL2_DRAFT_FP8, DFL2_HEAD2)

        dfl2_opt.draft_loaded(model, self.vllm_config)
        return model

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 so the walk and the rejection that checks it read the same
        # distribution; -inf because the cache kernel writes only the K
        # candidates.
        return torch.float32, -float("inf")

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        scores = self._score_candidates(candidate_ids, unary_logits, hidden_states)
        self.candidate_sampler.sample(
            candidate_ids,
            scores,
            num_reqs,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self.draft_logits,
            self.use_fp64_gumbel,
        )
        if self.enable_adaptive_verification:
            self._maybe_predict_acceptance(
                self.candidate_sampler.scores[:num_reqs].flatten(0, 1),
                self.sample_idx_mapping[:num_sample],
                self.sample_col[:num_sample],
            )

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = candidate_ids.shape[0]
        anchor_token_ids = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req : self.num_query_per_req
        ]
        return self.model.model.candidate_selector(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        )
