from keyword import iskeyword, kwlist

def _assert_valid_python_varname(name):
    if name.isidentifier() and not iskeyword(name):
        return
    raise ValueError(f"Received invalid Python variable name {name!r}. Avoid using the following names: {", ".join(kwlist)}")