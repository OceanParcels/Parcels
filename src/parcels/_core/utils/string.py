from keyword import iskeyword, kwlist

# def _assert_valid_python_varname(name):
#     try:
#         if name.isidentifier():
#             if not iskeyword(name):
#                 return
#             raise ValueError(f"Received invalid Python variable name {name!r}: it is a reserved keyword. Avoid using the following names: {', '.join(kwlist)}")
#         else:
#             raise ValueError(f"Received invalid Python variable name {name!r}: not a valid identifier.")
#     except Exception as e:
#         raise ValueError(f"Error validating variable name {name!r}: {e}")


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
