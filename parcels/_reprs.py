"""Parcels reprs"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parcels import Field, FieldSet, ParticleSet


def field_repr(field: Field) -> str:  # TODO v4: Rework or remove entirely
    """Return a pretty repr for Field"""
    out = f"""<{type(field).__name__}>
    name            : {field.name!r}
    data            : {field.data!r}
    extrapolate time: {field.allow_time_extrapolation!r}
"""
    return textwrap.dedent(out).strip()


def _format_list_items_multiline(items: list[str], level: int = 1) -> str:
    """Given a list of strings, formats them across multiple lines.

    Uses indentation levels of 4 spaces provided by ``level``.

    Example
    -------
    >>> output = _format_list_items_multiline(["item1", "item2", "item3"], 4)
    >>> f"my_items: {output}"
    my_items: [
        item1,
        item2,
        item3,
    ]
    """
    if len(items) == 0:
        return "[]"

    assert level >= 1, "Indentation level >=1 supported"
    indentation_str = level * 4 * " "
    indentation_str_end = (level - 1) * 4 * " "

    items_str = ",\n".join([textwrap.indent(i, indentation_str) for i in items])
    return f"[\n{items_str}\n{indentation_str_end}]"


def particleset_repr(pset: ParticleSet) -> str:
    """Return a pretty repr for ParticleSet"""
    if len(pset) < 10:
        particles = [repr(p) for p in pset]
    else:
        particles = [repr(pset[i]) for i in range(7)] + ["..."]

    out = f"""<{type(pset).__name__}>
    fieldset   :
{textwrap.indent(repr(pset.fieldset), " " * 8)}
    ptype      : {pset._ptype}
    # particles: {len(pset)}
    particles  : {_format_list_items_multiline(particles, level=2)}
"""
    return textwrap.dedent(out).strip()


def fieldset_repr(fieldset: FieldSet) -> str:  # TODO v4: Rework or remove entirely
    """Return a pretty repr for FieldSet"""
    fields_repr = "\n".join([repr(f) for f in fieldset.fields.values()])

    out = f"""<{type(fieldset).__name__}>
    fields:
{textwrap.indent(fields_repr, 8 * " ")}
"""
    return textwrap.dedent(out).strip()


def default_repr(obj: Any):
    if is_builtin_object(obj):
        return repr(obj)
    return object.__repr__(obj)


def is_builtin_object(obj):
    return obj.__class__.__module__ == "builtins"
