import sys

sys.path.insert(0, "/home/cdaunt/code/bosdi/src")

from bosdi.va import compile_va_unopt_with_split, lower
from bosdi.va.emitter import emit_source
from bosdi.va.va_defaults import parse_va_defaults_expanded

PSP103_VA = "/home/cdaunt/code/circulax/circulax/tests/data/va/psp103v4/psp103.va"
dump = compile_va_unopt_with_split(PSP103_VA)
defaults = parse_va_defaults_expanded(PSP103_VA)

# TYPE-only static (works correctly):
dev_type_only = lower(dump.modules[0], va_defaults=defaults, collapse_nodes=True,
                      static_params={"TYPE": 1}, class_name="PSP103N_TypeOnly")

# All-static (bug: sentinel values leak):
# static_params requires int|float values, not string literals; skip string-typed params.
def _to_python_value(spec):
    if spec.type_ == "float":
        return float(spec.default)
    if spec.type_ == "int":
        return int(spec.default)
    return None  # skip string params

all_static = {name: _to_python_value(spec) for name, spec in defaults.items()
              if _to_python_value(spec) is not None}
dev_all_static = lower(dump.modules[0], va_defaults=defaults, collapse_nodes=True,
                       static_params=all_static, class_name="PSP103N_AllStatic")

type_only_src = emit_source([dev_type_only])
all_static_src = emit_source([dev_all_static])

print(f"TYPE-only source length: {len(type_only_src):,} chars")
print(f"All-static source length: {len(all_static_src):,} chars")

bugs_found = 0
for sentinel in ["8e22", "8e+22", "1e22", "1e+22"]:
    count = all_static_src.count(sentinel)
    if count > 0:
        print(f"BUG: sentinel {sentinel!r} found {count} times in all-static emitted source")
        bugs_found += count

if bugs_found == 0:
    print("OK: no sentinel values in all-static emitted source")
else:
    print(f"FAIL: {bugs_found} total sentinel occurrences")
    sys.exit(1)
