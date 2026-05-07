"""Patch _build_nb09.py: replace all \"\"\" docstrings inside cell sources with # comments."""
import re

content = open("_build_nb09.py", encoding="utf-8").read()

# ── Replace normalize_per_window ''' docstring (already single-quote) → comment ──
content = content.replace(
    "    '''Instance normalization: setiap sample dinormalisasi oleh mean/std window-nya sendiri.\n"
    "    X: (n_samples, tau) - tiap baris adalah window tau lag\n"
    "    '''",
    "    # Instance normalization: setiap sample dinormalisasi oleh mean/std window-nya sendiri.\n"
    "    # X: (n_samples, tau) - tiap baris adalah window tau lag"
)

# ── Replace nb08_seed_chromosome multi-line \"\"\" docstring → comments ─────────
old_doc = (
    '    """\n'
    '    Kromosom initial solution dari config terbaik NB08, ditranslasi ke search space NB09.\n'
    '\n'
    '    NB08: filters=16, kernel_size=3, dense_units=16, dropout=0.25, lr=5e-4, batch_size=16\n'
    '    NB09 mapping:\n'
    '      filters=16        \u2192 idx 0  (in [16, 32, 64])\n'
    '      kernel_size=3     \u2192 idx 1  (in [2, 3])\n'
    '      pool_size=2       \u2192 idx 0  (satu-satunya opsi [2])\n'
    '      dense_units=32    \u2192 idx 0  (NB08:16 \u2192 nearest in [32, 64, 128])\n'
    '      dropout=0.3       \u2192 idx 2  (NB08:0.25 \u2192 nearest in [0.1, 0.2, 0.3])\n'
    '      learning_rate=0.001 \u2192 idx 0  (NB08:5e-4 \u2192 nearest in [0.001, 0.005])\n'
    '      batch_size=16     \u2192 idx 0  (in [16, 32])\n'
    '    """'
)
new_doc = (
    '    # Kromosom initial solution dari config terbaik NB08, ditranslasi ke search space NB09.\n'
    '    # NB08: filters=16, kernel_size=3, dense_units=16, dropout=0.25, lr=5e-4, batch_size=16\n'
    '    # NB09 mapping: filters=16(idx0), kernel_size=3(idx1), pool_size=2(idx0),\n'
    '    #   dense_units=32(idx0), dropout=0.3(idx2), learning_rate=0.001(idx0), batch_size=16(idx0)'
)
content = content.replace(old_doc, new_doc)

# ── Replace remaining single-line \"\"\" docstrings inside cell code → # comment ──
# These are lines like:     """Some description."""
def repl_docstring(m):
    indent = m.group(1)
    text   = m.group(2)
    return f"{indent}# {text}"

content = re.sub(r'(    )"""(.+?)"""', repl_docstring, content)

open("_build_nb09.py", "w", encoding="utf-8").write(content)

# Verify
remaining = [(i+1, l) for i, l in enumerate(content.splitlines()) if '"""' in l]
print(f"Remaining triple-double-quote lines: {len(remaining)}")
for ln, txt in remaining:
    print(f"  L{ln}: {txt[:90]}")
print("Done.")
