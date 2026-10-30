"""Constructor validation embedded by regenerate.py (no runtime file dependency)."""

# ruff: noqa: CPY001, F821, ANN001, ANN002, ANN003, ANN202

import inspect
from numbers import Real

_gan_init = AngelovGaN.__init__
_gan_signature = inspect.signature(_gan_init)


def _gan_checked_init(self, *args, **kwargs):
    bound = _gan_signature.bind(self, *args, **kwargs)
    for name in ("Rg", "Rd", "Rs", "Ri", "Rgd", "Rc", "Rcin", "Rdel", "Cdel", "Rth"):
        value = bound.arguments.get(name, _gan_signature.parameters[name].default)
        if isinstance(value, Real) and value <= 0:
            message = f"AngelovGaN requires {name} > 0 for its retained finite-resistance topology"
            raise ValueError(message)
    for name, allowed in (("Idsmod", range(5)), ("Capmod", range(5)), ("Igmod", range(2)), ("Selft", range(2))):
        value = bound.arguments.get(name, _gan_signature.parameters[name].default)
        if isinstance(value, Real) and value not in allowed:
            message = f"AngelovGaN {name} must be one of {tuple(allowed)}"
            raise ValueError(message)
    _gan_init(self, *args, **kwargs)


_gan_checked_init.__signature__ = _gan_signature
AngelovGaN.__init__ = _gan_checked_init
