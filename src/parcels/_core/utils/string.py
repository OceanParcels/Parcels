from keyword import iskeyword, kwlist

def _assert_str_and_python_varname(name):
    if not isinstance(name, str):
        raise TypeError(f"Expected a string for variable name, got {type(name).__name__} instead.")

    if not name.isidentifier():
        raise ValueError(
            f"Received invalid Python variable name {name!r}: not a valid identifier. "
            f"HINT: avoid using spaces, special characters, and starting with a number."
        )
    if iskeyword(name):
        raise ValueError(
            f"Received invalid Python variable name {name!r}: it is a reserved keyword. "
            f"HINT: avoid using the following names: {', '.join(kwlist)}"
        )