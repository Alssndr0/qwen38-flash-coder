"""Scheduler for the coder kit: vLLM's AsyncScheduler plus a step tracer and a live prefill-pacing policy.

Loaded with `--scheduler-cls mbx_sched.SmoothScheduler`. With the policy off it schedules exactly like the
AsyncScheduler it extends.

Step tracer (env MBX_STEP_TRACE=<dir>, or "trace": true in the live file): one JSON line per engine step in
<dir>/steps-<start>-<pid>.jsonl with the wall time it was scheduled (ts) and finished (te, when its output reached
the scheduler), how many requests decoded and how many tokens they verified, and the prefill chunks (tokens and
context end of each). Wall clock (time.time), so it lines up with the client recorder in bench/.

Pacing policy (live file MBX_SMOOTH_FILE, default /cache/mbx-smooth.json, re-read when it changes):
  {"mode": "off"}                      the AsyncScheduler as is
  {"mode": "pace", "cap": 1024, "gap": 2, "solo_cap": 0, "budget_ms": 0, ...}
    while any request is decoding:
      cap        at most this many prefill tokens per step (over all prefilling requests)
      gap        after a step that carried prefill, this many decode-only steps before the next prefill chunk
      budget_ms  when > 0: the chunk is sized from the cost model so the step lasts about budget_ms
                 (cost model: ms = a + b * prefill_tokens + c * prefill_tokens * context / 1e3, keys "a","b","c")
      min_chunk  smallest chunk worth a step (budget mode), default 256
      share      when > 0: time-share pacing instead of a fixed gap. After a prefill-carrying step that took Tp,
                 decode-only steps run until they have taken Tp * (1 - share) / share, so prefill gets about `share`
                 of the GPU time while others decode
    when nothing is decoding: solo_cap (0 = vLLM's own budget)
  {"mode": "slice", "min_tokens": 128, "chunk": 6912, "slice_ms": 40, "la": 2.4, "lb": 0.0071}
    layer-sliced prefill (worker side: mbx_sidecar.py, needs MBX_SIDECAR=1): while any request is decoding, a
    newcomer is admitted with its whole prompt allocated, its tokens are taken back out of the step, and the worker
    prefills them in chunks of `chunk` tokens, `slice_ms` of layers per step (per-layer cost la + lb * tokens ms),
    after each decode step. New arrivals wait while a sliced prefill runs. When the worker reports the bulk done,
    the prompt's last token is scheduled normally. Prompts under min_tokens new tokens prefill normally.
    "gap_ms" (with _busy / _late / late_s) sizes each slice to the decoders' gap target instead of slice_ms;
    "tlog": true logs every slice's GPU and CPU ms (worker log), the data the per-layer cost model is fitted on.
"""
import json
import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.request import RequestStatus

logger = init_logger("vllm." + __name__)

_LIVE = os.environ.get("MBX_SMOOTH_FILE", "/cache/mbx-smooth.json")
_TRACE_DIR = os.environ.get("MBX_STEP_TRACE", "")


class SmoothScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cfg = {"mode": "off"}
        self._cfg_mtime = None
        self._cfg_checked = 0.0
        self._gap_left = 0
        self._credit = 0.0      # seconds of decode-only time still owed (share mode)
        self._pf_inflight = 0   # prefill-carrying steps scheduled but not finished (share mode)
        self._last_te = None
        self._inflight = {}
        self._buf = []
        self._flushed = time.time()
        self._trace_file = None
        self._step_no = 0
        self._spec_k = self.num_spec_tokens
        self._orig_budget = self.max_num_scheduled_tokens
        self._orig_lpt = self.scheduler_config.long_prefill_token_threshold
        self._sc_job = None
        self._dec_ms = {}         # decoding streams -> EWMA of a decode-only step (ms), measured here
        self._sc_finishing = {}   # id(scheduler output) -> request ids whose first token that step's output carries
        self._sc_layers = self.vllm_config.model_config.hf_text_config.num_hidden_layers
        self._sc_seq = int(time.time() * 1000) % 1000000000
        self._read_cfg(force=True)
        logger.info("SmoothScheduler: live file %s, trace dir %s, policy %s", _LIVE, _TRACE_DIR or "-", self._cfg)

    # ------------------------------------------------------------------ live config
    def _read_cfg(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._cfg_checked < 0.5:
            return
        self._cfg_checked = now
        try:
            m = os.stat(_LIVE).st_mtime
        except OSError:
            return
        if m == self._cfg_mtime:
            return
        try:
            with open(_LIVE) as f:
                cfg = json.load(f)
            self._cfg, self._cfg_mtime = cfg, m
            self._gap_left = 0
            self._credit = 0.0
            self._pf_inflight = 0
            logger.info("SmoothScheduler: policy %s", cfg)
        except Exception as e:  # half-written file: keep the old policy
            logger.warning("SmoothScheduler: cannot read %s (%s)", _LIVE, e)

    def _tracing(self) -> bool:
        return bool(_TRACE_DIR) or bool(self._cfg.get("trace"))

    # ------------------------------------------------------------------ policy
    def _chunk_for_budget(self, cfg) -> int:
        """Prefill tokens that keep the step near budget_ms, from the linear cost model."""
        budget = float(cfg.get("budget_ms", 0))
        a, b, c = float(cfg.get("a", 60.0)), float(cfg.get("b", 0.2)), float(cfg.get("c", 0.0))
        ctx = 0
        for r in self.running:
            if r.is_prefill_chunk:
                ctx = max(ctx, r.num_computed_tokens)
        per_tok = b + c * ctx / 1e3
        n = int(max(0.0, budget - a) / max(per_tok, 1e-6))
        return max(int(cfg.get("min_chunk", 256)), min(n, self._orig_budget))

    # ------------------------------------------------------------------ slice mode
    def _unschedule(self, out, req, n: int) -> None:
        """Take a prefill chunk back out of this step (blocks stay allocated; the worker still learns about the
        request through scheduled_new_reqs / scheduled_cached_reqs); the same chunk is planned again later."""
        del out.num_scheduled_tokens[req.request_id]
        out.total_num_scheduled_tokens -= n
        req.num_computed_tokens -= n
        req.num_in_flight_tokens -= n
        req.is_prefill_chunk = True
        self._inflight_prefills.add(req)

    def _slice_budget(self, cfg, reqs, n_dec: int = 0) -> float:
        """Prefill ms for this step. With "gap_ms": the decoding streams' gap target minus the measured decode step
        for this many streams (so few streams leave prefill more room at the same smoothness); else "slice_ms".
        Progressive: more when others wait behind this prefill (busy), more again once one of its requests has waited
        longer than late_s (late)."""
        mine = {rid for rid, *_ in reqs}
        backlog = len(self.waiting) + sum(1 for r in self.running if r.is_prefill_chunk and r.request_id not in mine)
        now = time.time()
        waited = max((now - (getattr(self.requests.get(rid), "arrival_time", None) or now)) for rid in mine)
        late = waited > float(cfg.get("late_s", 5.0))
        if "gap_ms" in cfg:
            gap = float(cfg["gap_ms"])
            if backlog:
                gap = max(gap, float(cfg.get("gap_ms_busy", gap)))
            if late:
                gap = max(gap, float(cfg.get("gap_ms_late", gap)))
            dec = self._dec_ms.get(n_dec) or (30.0 + 13.0 * n_dec)
            return max(float(cfg.get("slice_min_ms", 40)), gap - dec)
        ms = float(cfg.get("slice_ms", 80))
        if backlog:
            ms = max(ms, float(cfg.get("slice_ms_busy", 130)))
        if late:
            ms = max(ms, float(cfg.get("slice_ms_late", 200)))
        return ms

    def _plan_slice(self, cfg, decoding: bool, n_dec: int = 0) -> dict:
        """This step's layers for the current job, from the per-layer cost model ms = la + lb * tokens."""
        job = self._sc_job
        lo = job["layer"]
        if not decoding and not cfg.get("cap_idle", 1):
            hi = self._sc_layers  # nobody to protect: finish now
        else:
            per_layer = float(cfg.get("la", 2.4)) + float(cfg.get("lb", 0.0071)) * job["n"]
            budget = self._slice_budget(cfg, job["reqs"], n_dec) if decoding else float(cfg.get("idle_ms", 600))
            k = max(1, int(budget // per_layer))
            hi = min(self._sc_layers, lo + k)
        job["layer"] = hi
        msg = {"seq": job["seq"], "reqs": job["reqs"], "lo": lo, "hi": hi}
        for k in ("prof", "debug", "tlog"):
            if cfg.get(k):
                msg[k] = cfg[k]
        return msg

    def _schedule_slice(self, throttle_prefills: bool, cfg):
        """Per-chunk interception: the normal scheduler picks and allocates prefill chunks as usual; when someone is
        decoding they are taken back out of the step and the worker computes them a few layers per step (one job
        for all chunks of the step). While a job runs, every request still prefilling is held and new admissions
        wait, so steps carry decodes only. The step whose slice reaches the last layer also samples: the prompts
        that end there get their first token in that step's output (merged by the worker) and decode from the next
        step on; the others continue with their next chunk."""
        job = self._sc_job
        if job is not None:
            live = [r for r in job["reqs"] if (q := self.requests.get(r[0])) is not None and not q.is_finished()
                    and q in self.running]
            if not live:
                logger.info("SmoothScheduler: sliced job %s abandoned (requests gone)", job["seq"])
                self._sc_job = job = None
            else:
                job["reqs"] = live
        if job is not None:
            mine = {r[0] for r in job["reqs"]}
            n_dec = sum(1 for r in self.running if not r.is_prefill_chunk and r.request_id not in mine)
            decoding = n_dec > 0
            held = [r for r in self.running if r.request_id in mine or r.is_prefill_chunk]
            self.running[:] = [r for r in self.running if not (r.request_id in mine or r.is_prefill_chunk)]
            paused = self._pause_state
            if paused == PauseState.UNPAUSED:
                self._pause_state = PauseState.PAUSED_NEW
            try:
                out = AsyncScheduler.schedule(self, throttle_prefills)
            finally:
                self._pause_state = paused
                self.running.extend(r for r in held if r not in self.running)
            msg = self._plan_slice(cfg, decoding or bool(cfg.get("force")), n_dec)
            out.mbx_sidecar = msg
            if msg["hi"] >= self._sc_layers:
                self._sc_finishing[id(out)] = [r[0] for r in job["reqs"] if r[3]]
                self._sc_job = None
            return out, False
        decoding = any(not r.is_prefill_chunk for r in self.running) or bool(cfg.get("force"))
        out = AsyncScheduler.schedule(self, throttle_prefills)
        if not decoding:
            return out, False
        reqs = []
        for req_id, n in list(out.num_scheduled_tokens.items()):
            req = self.requests.get(req_id)
            if req is None:
                continue
            before = req.num_computed_tokens - n
            if before >= req.num_prompt_tokens or req.num_output_tokens > 0:
                continue  # decoding (or a resumed request replaying output tokens)
            sp = getattr(req, "sampling_params", None)
            if (req.has_encoder_inputs or req.use_structured_output
                    or (sp is not None and (sp.prompt_logprobs is not None or sp.logprobs is not None))):
                continue  # can't be sliced (images, grammars, logprobs): prefills normally, never starves
            if n < int(cfg.get("min_tokens", 16)):
                continue  # tiny: rides along like a decode
            del out.num_scheduled_tokens[req_id]
            out.total_num_scheduled_tokens -= n
            req.num_in_flight_tokens -= n
            reqs.append([req_id, before, before + n, before + n >= req.num_tokens])
        if reqs:
            n_dec = sum(1 for r in self.running if not r.is_prefill_chunk and r.request_id not in {x[0] for x in reqs})
            self._sc_seq += 1
            self._sc_job = {"seq": self._sc_seq, "reqs": reqs, "layer": 0, "n": sum(e - s for _, s, e, _ in reqs),
                            "t0": time.time()}
            msg = self._plan_slice(cfg, True, n_dec)
            out.mbx_sidecar = msg
            if msg["hi"] >= self._sc_layers:
                self._sc_finishing[id(out)] = [r[0] for r in reqs if r[3]]
                self._sc_job = None
            if cfg.get("debug"):
                logger.info("SmoothScheduler: sliced job %s: %s", self._sc_seq, reqs)
        return out, False

    def schedule(self, throttle_prefills: bool = False):
        self._read_cfg()
        cfg = self._cfg
        mode = cfg.get("mode", "off")
        decoding = [r for r in self.running if not r.is_prefill_chunk]
        has_prefill = bool(self.waiting) or any(r.is_prefill_chunk for r in self.running)
        restore = False
        decode_only = False
        if not has_prefill or not decoding:
            self._credit = 0.0
            self._pf_inflight = 0
        if mode == "pace" and has_prefill:
            if decoding:
                share = float(cfg.get("share", 0))
                if share > 0 and (self._credit > 0 or self._pf_inflight > 0):
                    decode_only = True
                elif share <= 0 and self._gap_left > 0:
                    decode_only = True
                    self._gap_left -= 1
                else:
                    cap = int(cfg.get("cap", 0))
                    if float(cfg.get("budget_ms", 0)) > 0:
                        bcap = self._chunk_for_budget(cfg)
                        cap = min(cap, bcap) if cap > 0 else bcap
                    if cap > 0:
                        dec_tokens = len(decoding) * (1 + self._spec_k)
                        self.max_num_scheduled_tokens = min(self._orig_budget, dec_tokens + cap)
                        self.scheduler_config.long_prefill_token_threshold = cap
                        restore = True
            else:
                solo = int(cfg.get("solo_cap", 0))
                if solo > 0:
                    self.max_num_scheduled_tokens = min(self._orig_budget, solo)
                    self.scheduler_config.long_prefill_token_threshold = solo
                    restore = True
        try:
            if mode == "slice":
                out, _ = self._schedule_slice(throttle_prefills, cfg)
            elif decode_only:
                self.prefill_capacity_bound = False
                out = super().schedule(True)
            else:
                out = super().schedule(throttle_prefills)
        finally:
            if restore:
                self.max_num_scheduled_tokens = self._orig_budget
                self.scheduler_config.long_prefill_token_threshold = self._orig_lpt

        pf_tokens, pf = 0, []
        dec_n = dec_t = 0
        for req_id, n in out.num_scheduled_tokens.items():
            req = self.requests.get(req_id)
            if req is None:
                continue
            before = req.num_computed_tokens - n
            if before < req.num_prompt_tokens:
                pf_tokens += n
                pf.append((n, before + n))
            else:
                dec_n += 1
                dec_t += n
        if mode == "pace" and pf_tokens and dec_n:
            self._gap_left = int(cfg.get("gap", 0))
            self._pf_inflight += 1

        if out.total_num_scheduled_tokens > 0:
            self._step_no += 1
            self._inflight[id(out)] = {
                "i": self._step_no, "ts": time.time(), "dn": dec_n, "dt": dec_t, "pt": pf_tokens,
                "pf": pf, "run": len(self.running), "wait": len(self.waiting), "do": int(decode_only),
                "sc": self._sc_job["seq"] if self._sc_job else 0,
            }
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        rec = self._inflight.pop(id(scheduler_output), None)
        fin = self._sc_finishing.pop(id(scheduler_output), None)
        if fin:
            # a sliced prompt ended in this step: its first token is in the worker's output; count the request as
            # scheduled here so the base bookkeeping (output, stop checks, cache, placeholders) handles it
            for rid in fin:
                req = self.requests.get(rid)
                if (req is not None and not req.is_finished() and rid in model_runner_output.req_id_to_index
                        and rid not in scheduler_output.num_scheduled_tokens):
                    scheduler_output.num_scheduled_tokens[rid] = 1
                    req.num_in_flight_tokens += 1
                elif req is not None and not req.is_finished():
                    logger.error("SmoothScheduler: no first token from the worker for %s", rid)
        res = super().update_from_output(scheduler_output, model_runner_output)
        if rec is not None:
            te = time.time()
            rec["te"] = te
            dur = te - max(rec["ts"], self._last_te or rec["ts"])
            self._last_te = te
            if rec["dn"] and not rec["pt"] and not rec.get("sc"):
                d = self._dec_ms.get(rec["dn"])
                ms = 1000.0 * dur
                self._dec_ms[rec["dn"]] = ms if d is None else 0.9 * d + 0.1 * ms
            share = float(self._cfg.get("share", 0))
            if share > 0 and self._cfg.get("mode") == "pace":
                if rec["pt"] and rec["dn"]:
                    self._pf_inflight = max(0, self._pf_inflight - 1)
                    self._credit += dur * (1.0 - share) / share
                elif rec["do"]:
                    self._credit -= dur
            if self._tracing():
                self._buf.append(rec)
                if te - self._flushed > 1.0 or len(self._buf) > 500:
                    self._flush()
        return res

    def _flush(self) -> None:
        self._flushed = time.time()
        if not self._buf:
            return
        try:
            if self._trace_file is None:
                d = _TRACE_DIR or os.path.join(os.path.dirname(_LIVE), "steptrace")
                os.makedirs(d, exist_ok=True)
                self._trace_file = open(os.path.join(d, f"steps-{int(time.time())}-{os.getpid()}.jsonl"), "a")
            self._trace_file.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in self._buf))
            self._trace_file.flush()
        except Exception as e:
            logger.warning("SmoothScheduler: trace write failed (%s)", e)
        self._buf = []
