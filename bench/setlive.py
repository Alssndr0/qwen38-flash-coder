"""Set keys in cache/mbx-smooth.json (the live slicing policy): setlive.py key=value ... (value JSON)."""
import json, os, sys
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "mbx-smooth.json")
d = json.load(open(p))
for a in sys.argv[1:]:
    k, v = a.split("=", 1)
    if v == "-":
        d.pop(k, None)
    else:
        d[k] = json.loads(v)
json.dump(d, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
print(d)
