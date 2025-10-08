# Generic Python helpers


def isinstance_noimport(obj, class_or_tuple):
    """A version of isinstance that does not require importing the class.
    This is useful to avoid circular imports.
    """
    return (
        type(obj).__name__ == class_or_tuple
        if isinstance(class_or_tuple, str)
        else type(obj).__name__ in class_or_tuple
    )


def test_isinstance_noimport():
    class A:
        pass

    class B:
        pass

    a = A()
    b = B()

    assert isinstance_noimport(a, "A")
    assert not isinstance_noimport(a, "B")
    assert isinstance_noimport(b, ("A", "B"))
    assert not isinstance_noimport(b, "C")
