import pytest

from parcels._compat import add_note


def test_add_note_and_raise_value_error():
    with pytest.raises(ValueError) as excinfo:
        try:
            raise ValueError("original message")
        except ValueError as e:
            e = add_note(e, "additional note")
            raise e
    assert "additional note" in str(excinfo.value)
    assert "original message" in str(excinfo.value)
