#!/usr/bin/env python3
"""vLLM #64675 port check: the patched qwen3 arg converter (files/v34/vllm/parser/qwen3.py) must give exactly the
image's output on every streamed prefix, with one cache per parser carried across prefixes and across calls.

Run inside the serve container (CPU only). OLD is the image's own copy (the kit now mounts the patched one over it):
  docker run --rm --entrypoint cat myllmbox/qwen38-flash-next-vllm:v5.4 \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/qwen3.py > /tmp/qwen3_old.py
  docker cp /tmp/qwen3_old.py qwen38-flash-next:/tmp/ && docker cp files/v34/vllm/parser/qwen3.py qwen38-flash-next:/tmp/qwen3_v34.py
  docker cp bench/parser34_test.py qwen38-flash-next:/tmp/ && docker cp DATA qwen38-flash-next:/tmp/replay.jsonl
  docker exec qwen38-flash-next python3 /tmp/parser34_test.py /tmp/qwen3_old.py /tmp/qwen3_v34.py /tmp/replay.jsonl
"""
import importlib.util
import json
import random
import sys
import time



def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old = load("qwen3_old", sys.argv[1])
new = load("qwen3_v34", sys.argv[2])


def render(args):
    """Arguments as the chat template writes them after <function=NAME>."""
    out = "\n"
    for k, v in args.items():
        val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        out += f"<parameter={k}>\n{val}\n</parameter>\n"
    return out


calls = []
seen = set()
for line in open(sys.argv[3]):
    body = json.loads(line)["body"]
    for m in body.get("messages", []):
        for tc in m.get("tool_calls") or []:
            a = tc["function"]["arguments"]
            a = json.loads(a) if isinstance(a, str) else a
            raw = render(a)
            if raw not in seen:
                seen.add(raw)
                calls.append(raw)
n_real = len(calls)
# Edge cases: closed by the next opener, spaced tags, angle brackets and tags inside values, repeated keys, empty
# values, no trailing newline, junk between parameters, an unterminated last value.
calls += [
    "\n<parameter=a>\n1\n<parameter=b>\n2\n</parameter>\n",
    "\n< parameter = path >\n/x/y.py\n</ parameter >\n<parameter=content>\nif a < b and c > d:\n    pass\n</parameter>\n",
    "\n<parameter=content>\n<div><parameter>not a tag</parameter></div>\n</parameter>\n<parameter=x>\n1\n</parameter>",
    "\n<parameter=k>\nfirst\n</parameter>\n<parameter=k>\nsecond\n</parameter>\n",
    "\n<parameter=empty>\n</parameter>\n<parameter=e2></parameter>\n<parameter=z>\nz",
    "\njunk before\n<parameter=a>\nA\n</parameter>\nnoise<parameter=b>\nB\n</parameter>tail",
    "\n<parameter=cmd>\nls -la\n</parameter>\n<parameter=desc>\nunterminated value with < and >",
    "<parameter=a>x</parameter><parameter=b>y</parameter><parameter=c>z",
]
rng = random.Random(0)
bad = 0
checked = 0


def check(raw, closed, step):
    global bad, checked
    i = 0
    while True:
        i = min(len(raw), i + step())
        for partial in (True,) if i < len(raw) else (True, False):
            a = old._qwen3_arg_converter(raw[:i], partial)
            b = new._qwen3_arg_converter(raw[:i], partial, closed)
            checked += 1
            if a != b:
                bad += 1
                if bad <= 5:
                    print("MISMATCH", repr(raw[:i][-80:]), partial, a[:120], b[:120])
        if i >= len(raw):
            break


for raw in calls:  # every character, a fresh cache per call
    check(raw, new._ClosedParams(), lambda: 1)
for raw in calls:  # engine-like chunks of 1-12 characters
    check(raw, new._ClosedParams(), lambda: rng.randint(1, 12))
shared = new._ClosedParams()  # one parser streaming many calls in a row (a response with several tool calls)
for raw in rng.sample(calls, 300):
    check(raw, shared, lambda: rng.randint(1, 12))
print(f"calls {len(calls)} ({n_real} unique recorded + {len(calls) - n_real} edge), conversions {checked}, mismatches {bad}")


def cost(raw, step=16):
    t = time.perf_counter()
    for i in range(step, len(raw), step):
        old._qwen3_arg_converter(raw[:i], True)
    t_old = time.perf_counter() - t
    closed = new._ClosedParams()
    t = time.perf_counter()
    for i in range(step, len(raw), step):
        new._qwen3_arg_converter(raw[:i], True, closed)
    return t_old, time.perf_counter() - t


big = render({"filePath": "/repo/src/big.py", "content": "".join(f"line {i}: x = {i} * 2\n" for i in range(3000))})
loop = "\n" + "".join(f"<parameter=p{i % 2}>\nvalue {i}\n</parameter>\n" for i in range(400))
many = render({f"arg{i}": f"value number {i}" for i in range(200)})
for name, raw in (("one 64K-char value (file write)", big), ("loop, 400 closed params", loop),
                  ("200 closed params", many)):
    to, tn = cost(raw)
    print(f"cost {name}: {len(raw)} chars, streamed every 16 chars: old {to * 1e3:.0f} ms, new {tn * 1e3:.0f} ms")
sys.exit(1 if bad else 0)
