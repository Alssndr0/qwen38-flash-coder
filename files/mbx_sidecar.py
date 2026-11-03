"""Layer-sliced prefill (worker side). While other requests decode, a newcomer's prompt chunk is prefilled a few decoder
layers per engine step, right after the normal decode step, so the decoding streams keep getting a token every step
instead of freezing for a whole prefill step (a 6,912-token chunk is one ~2.5 s step on this model).

Protocol (the scheduler, mbx_sched.SmoothScheduler mode "slice", plans everything):
  - The scheduler picks and allocates prefill chunks as usual, takes them back out of the step, and sends one slice per
    step on the SchedulerOutput: `so.mbx_sidecar = {"seq", "reqs": [[req_id, start, end, final], ...], "lo", "hi"}`.
  - After the step's own sampling, the worker runs layers [lo, hi) of those chunks through the runner's own
    execute_model on a synthetic SchedulerOutput holding only them (eager, uncompiled), so inputs, attention / GDN /
    PLE metadata, mamba align state copies and slot mappings are built exactly as for a normal prefill chunk. The model
    forward (v54/model.py `_forward_slice`) keeps the HC carry and the DFlash2 capture states between slices.
  - The slice that reaches the last layer is a full step for the synthetic batch: sampling, computed-token and state
    updates, the drafter's context K/V and drafts. A prompt's first token is merged into that step's own output, so the
    scheduler receives it like any prefill's first token, and the request decodes from the next step.
Exactness: every layer of every chunk runs once, on the same chunk the scheduler planned, with the same kernels; checked
teacher-forced (bench/exact2.py, exact3.py) including later turns that reuse a sliced turn's prefix cache.
"""
import dataclasses
import time

import torch

from vllm.logger import init_logger
from vllm.v1.outputs import AsyncModelRunnerOutput

logger = init_logger("vllm." + __name__)

SLICE = {"on": False, "lo": 0, "hi": 0, "stash": None}
_ORIG_SAMPLE = None
_installed = False


def install() -> None:
    """Wrap the V2 GPU model runner (once per process; the hooks only act in the worker)."""
    global _installed, _ORIG_SAMPLE
    if _installed:
        return
    _installed = True
    import vllm.v1.worker.gpu.model_runner as mr

    orig_exec = mr.GPUModelRunner.execute_model
    orig_sample = mr.GPUModelRunner.sample_tokens
    _ORIG_SAMPLE = orig_sample
    orig_dispatch = mr.dispatch_cg_and_sync_dp
    orig_sfc = mr.set_forward_context

    def dispatch(*a, **k):
        if SLICE["on"]:  # slices run eager: the compiled graphs hold the full-model forward
            k["need_eager"] = True
            if "allow_ubatching" in k:
                k["allow_ubatching"] = False
        return orig_dispatch(*a, **k)

    def sfc(*a, **k):
        if SLICE["on"]:
            k["skip_compiled"] = True
        return orig_sfc(*a, **k)

    mr.dispatch_cg_and_sync_dp = dispatch
    mr.set_forward_context = sfc

    def execute_model(self, scheduler_output, *a, **k):
        dummy = k.get("dummy_run", a[1] if len(a) > 1 else False)
        if SLICE["on"] or dummy:
            return orig_exec(self, scheduler_output, *a, **k)
        sc = _sidecar(self)
        sc.receive(scheduler_output)
        out = orig_exec(self, scheduler_output, *a, **k)
        if scheduler_output.total_num_scheduled_tokens == 0 and sc.cur is not None:
            # nothing else in this step (no sample_tokens call follows): run the slice now
            sc.step()
            if sc.merge is not None:
                out = _merge_now(out, sc.take_merge())
        return out

    def sample_tokens(self, grammar_output, *a, **k):
        out = orig_sample(self, grammar_output, *a, **k)
        if SLICE["on"]:
            return out
        sc = _sidecar(self)
        sc.step()
        if sc.merge is not None:
            out = MergedOutput(out, sc.take_merge())
        return out

    mr.GPUModelRunner.execute_model = execute_model
    mr.GPUModelRunner.sample_tokens = sample_tokens
    logger.info("mbx_sidecar: layer-sliced prefill hooks installed")


def _sidecar(runner) -> "Sidecar":
    sc = getattr(runner, "_mbx_sidecar", None)
    if sc is None:
        sc = Sidecar(runner)
        runner._mbx_sidecar = sc
    return sc


def _resolve(out):
    return out.get_output() if isinstance(out, AsyncModelRunnerOutput) else out


def _append_tokens(mo, syn, req_ids):
    """A copy of `mo` with the sampled tokens of req_ids (from the synthetic step's output `syn`) appended."""
    add = []
    for rid in req_ids:
        if syn is not None and rid in syn.req_id_to_index and syn.sampled_token_ids:
            toks = list(syn.sampled_token_ids[syn.req_id_to_index[rid]])
            if toks:
                add.append((rid, toks))
    if not add:
        return mo
    req_ids_new = list(mo.req_ids)
    index = dict(mo.req_id_to_index)
    sampled = [list(x) for x in (mo.sampled_token_ids or [])]
    sampled += [[] for _ in range(len(req_ids_new) - len(sampled))]
    for rid, toks in add:
        index[rid] = len(req_ids_new)
        req_ids_new.append(rid)
        sampled.append(toks)
    return dataclasses.replace(mo, req_ids=req_ids_new, req_id_to_index=index, sampled_token_ids=sampled)


def _merge_now(out, merge):
    syn_out, req_ids = merge
    return _append_tokens(_resolve(out), _resolve(syn_out), req_ids)


class MergedOutput(AsyncModelRunnerOutput):
    """The step's own async output plus the first tokens of prompts whose sliced prefill finished in this step."""

    def __init__(self, real, merge):
        self.real = real
        self.syn_out, self.req_ids = merge

    def get_output(self):
        return _append_tokens(_resolve(self.real), _resolve(self.syn_out), self.req_ids)


class Sidecar:
    def __init__(self, runner) -> None:
        self.r = runner
        self.nl = runner.vllm_config.model_config.hf_text_config.num_hidden_layers
        self.seq = None        # job being sliced
        self.stash: dict = {}  # HC carry + capture states between slices (rows in synthetic batch order)
        self.rows = None       # [(req_id, n)] of the stash rows
        self.cur = None        # this step's slice: {"seq", "reqs", "lo", "hi"}
        self.merge = None      # (synthetic output, [req ids whose first token it holds])
        self.num_spec = int(getattr(runner, "num_speculative_steps", 0) or 0) or 7
        self.timing = None
        self.slow_ms = 400
        self.tlog = False
        self.profiled = False

    def take_merge(self):
        m, self.merge = self.merge, None
        return m

    def receive(self, so) -> None:
        k = getattr(so, "num_spec_tokens_to_schedule", None)
        if k:
            self.num_spec = k  # synthetic steps draft like the real ones
        msg = getattr(so, "mbx_sidecar", None)
        self.cur = msg
        if msg is not None and msg["seq"] != self.seq:
            if msg["lo"] != 0:
                logger.error("mbx_sidecar: job %s starts at layer %d (missed its first slice)", msg["seq"], msg["lo"])
            self.seq, self.stash, self.rows = msg["seq"], {}, None

    def _synthetic_output(self, reqs):
        from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

        so = SchedulerOutput.make_empty()
        cr = CachedRequestData.make_empty()
        total = 0
        for rid, start, end, _final in reqs:
            n = end - start
            so.num_scheduled_tokens[rid] = n
            total += n
            cr.req_ids.append(rid)
            cr.new_block_ids.append(None)
            cr.num_computed_tokens.append(start)
            cr.num_output_tokens.append(0)
        so.total_num_scheduled_tokens = total
        so.scheduled_cached_reqs = cr
        so.num_spec_tokens_to_schedule = self.num_spec
        return so

    def _drop_rows(self, keep: set) -> None:
        """A member left mid-job (aborted): keep only the stash rows of the members still here."""
        if self.rows is None or not self.stash:
            return
        idx, off = [], 0
        for rid, n in self.rows:
            if rid in keep:
                idx.extend(range(off, off + n))
            off += n
        sel = torch.tensor(idx, dtype=torch.long, device=next(iter(v for v in self.stash.values() if v is not None
                                                                   and isinstance(v, torch.Tensor))).device)
        for key in ("h", "bo", "inj"):
            if isinstance(self.stash.get(key), torch.Tensor):
                self.stash[key] = self.stash[key].index_select(0, sel)
        if "aux" in self.stash:
            self.stash["aux"] = {k: v.index_select(0, sel) for k, v in self.stash["aux"].items()}
        self.rows = [(rid, n) for rid, n in self.rows if rid in keep]

    def step(self) -> None:
        msg, self.cur = self.cur, None
        if msg is None:
            return
        rs = self.r.req_states
        reqs = [r for r in msg["reqs"] if r[0] in rs.req_id_to_index]
        if len(reqs) != len(msg["reqs"]):
            self._drop_rows({r[0] for r in reqs})
        if not reqs:
            self.seq, self.stash, self.rows = None, {}, None
            return
        lo, hi = int(msg["lo"]), int(msg["hi"])
        self._report_timing()
        self.tlog = bool(msg.get("tlog"))
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        self.timing = (ev0, ev1, msg["seq"], [(r[0][-8:], r[2] - r[1], r[1]) for r in reqs], lo, hi, time.time())
        so = self._synthetic_output(reqs)
        import vllm.v1.worker.gpu.model_runner as mr
        import vllm.v1.worker.gpu.spec_decode.speculator as spm

        # the draft-depth controller (mbx_depth) keeps per-slot history of the real steps: it must not see these
        saved = (getattr(mr, "_mbx_depth", None), getattr(spm, "_mbx_depth", None))
        mr._mbx_depth = spm._mbx_depth = None
        try:
            SLICE.update(on=True, lo=lo, hi=hi, stash=self.stash)
            prof = None
            if msg.get("prof") and not self.profiled and hi - lo >= 4:
                self.profiled = True
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
                prof.__enter__()
            try:
                self.r.execute_model(so)
            finally:
                SLICE["on"] = False
                if prof is not None:
                    prof.__exit__(None, None, None)
                    tab = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
                    logger.warning("mbx_sidecar: CPU profile of one slice (layers %d..%d, %d tokens):\n%s", lo, hi,
                                   so.total_num_scheduled_tokens, tab)
            st = self.r.execute_model_state
            if st is not None and self.rows is None:
                n_of = dict(so.num_scheduled_tokens)
                self.rows = [(rid, n_of[rid]) for rid in st.input_batch.req_ids]
            if hi >= self.nl:
                # the last slice is a normal step for the synthetic batch (sampling, state updates, drafter);
                # the first tokens of the prompts that end here go into this step's own output
                syn = _ORIG_SAMPLE(self.r, None)
                finals = [r[0] for r in reqs if r[3]]
                if finals:
                    self.merge = (syn, finals)
                self.seq, self.stash, self.rows = None, {}, None
                return
            if st is not None:
                # as for any prefill chunk that samples nothing: neutral accepted count, no RecoverSSM commit
                self.r.model_state.postprocess_state(st.input_batch.idx_mapping, 0)
            self.r.execute_model_state = None
        finally:
            mr._mbx_depth, spm._mbx_depth = saved
            if self.timing is not None:
                self.timing[1].record()
                self.timing = self.timing[:6] + (time.time() - self.timing[6],)

    def _report_timing(self) -> None:
        """Log the previous slice when it was slow (its events are long done by the next step)."""
        t, self.timing = self.timing, None
        if t is None:
            return
        ev0, ev1, seq, reqs, lo, hi, cpu_s = t
        try:
            ms = ev0.elapsed_time(ev1)
        except RuntimeError:
            return
        if self.tlog:  # policy "tlog": every slice's GPU time (fits the planner's la / lb / lf)
            logger.info("mbx_sidecar: slice GPU %.1f ms CPU %.1f ms: job %s layers %d..%d of %d reqs %s", ms,
                        cpu_s * 1000.0, seq, lo, hi, self.nl, reqs)
        if ms > float(self.slow_ms) or cpu_s * 1000.0 > float(self.slow_ms):
            logger.warning("mbx_sidecar: slow slice: GPU %.0f ms, CPU %.0f ms: job %s layers %d..%d reqs (id, tokens, "
                           "start) %s", ms, cpu_s * 1000.0, seq, lo, hi, reqs)
