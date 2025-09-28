import numpy as np
import pytest

from parcels._core.utils.time import TimeInterval
from parcels._datasets.structured.generic import TIME
from parcels._core.particle import (
    _SAME_AS_FIELDSET_TIME_INTERVAL,
    Particle,
    ParticleClass,
    Variable,
    create_particle_data,
)


def test_variable_init():
    var = Variable("test")
    assert var.name == "test"
    assert var.dtype == np.float32
    assert var.to_write
    assert var.attrs == {}


def test_variable_invalid_init():
    with pytest.raises(ValueError, match="to_write must be one of .*\. Got to_write="):
        Variable("name", to_write="test")

    with pytest.raises(ValueError, match="to_write must be one of .*\. Got to_write="):
        Variable("name", to_write="test")

    for name in ["a b", "123", "while"]:
        with pytest.raises(ValueError, match="Particle variable has to be a valid Python variable name. Got "):
            Variable(name)

    with pytest.raises(ValueError, match="Attributes cannot be set if to_write=False"):
        Variable("name", to_write=False, attrs={"description": "metadata to write"})


@pytest.mark.parametrize(
    "variable, expected",
    [
        (
            Variable("test", np.float32, 0.0, True, {"some": "metadata"}),
            "Variable(name='test', dtype=dtype('float32'), initial=0.0, to_write=True, attrs={'some': 'metadata'})",
        ),
        (
            Variable("test", np.float32, 0.0, True),
            "Variable(name='test', dtype=dtype('float32'), initial=0.0, to_write=True, attrs={})",
        ),
    ],
)
def test_variable_repr(variable, expected):
    assert repr(variable) == expected


def test_particleclass_init():
    ParticleClass(
        variables=[
            Variable("vara", dtype=np.float32),
            Variable("varb", dtype=np.float32, to_write=False),
            Variable("varc", dtype=np.float32),
        ]
    )


def test_particleclass_invalid_vars():
    with pytest.raises(ValueError, match="All items in variables must be instances of Variable. Got"):
        ParticleClass(variables=[Variable("vara", dtype=np.float32), "not a variable class"])

    with pytest.raises(TypeError, match="Expected list of Variable objects, got "):
        ParticleClass(variables="not a list")


@pytest.mark.parametrize(
    "obj, expected",
    [
        (
            ParticleClass(
                variables=[
                    Variable("vara", dtype=np.float32, to_write=True),
                    Variable("varb", dtype=np.float32, to_write=False),
                    Variable("varc", dtype=np.float32, to_write=True),
                ]
            ),
            """ParticleClass(variables=[
    Variable(name='vara', dtype=dtype('float32'), initial=0, to_write=True, attrs={}),
    Variable(name='varb', dtype=dtype('float32'), initial=0, to_write=False, attrs={}),
    Variable(name='varc', dtype=dtype('float32'), initial=0, to_write=True, attrs={})
])""",
        ),
    ],
)
def test_particleclass_repr(obj, expected):
    assert repr(obj) == expected


def test_particleclass_add_variable():
    p_initial = ParticleClass(variables=[Variable("vara", dtype=np.float32)])
    variables = [
        Variable("varb", dtype=np.float32, to_write=True),
        Variable("varc", dtype=np.float32, to_write=False),
    ]
    p_final = p_initial.add_variable(variables)

    assert len(p_final.variables) == 3
    assert p_final.variables[0].name == "vara"
    assert p_final.variables[1].name == "varb"
    assert p_final.variables[2].name == "varc"


def test_particleclass_add_variable_in_loop():
    p = ParticleClass(variables=[Variable("vara", dtype=np.float32)])
    vars = [Variable("sample_var"), Variable("sample_var2")]
    p_loop = p
    for var in vars:
        p_loop = p_loop.add_variable(var)

    p_list = p.add_variable(vars)

    for var1, var2 in zip(p_loop.variables, p_list.variables, strict=True):
        assert var1.name == var2.name
        assert var1.dtype == var2.dtype
        assert var1.to_write == var2.to_write


def test_particleclass_add_variable_collision():
    p_initial = ParticleClass(variables=[Variable("vara", dtype=np.float32)])

    with pytest.raises(ValueError, match="Variable name already exists: "):
        p_initial.add_variable([Variable("vara", dtype=np.float32, to_write=True)])


@pytest.mark.parametrize(
    "particle",
    [
        ParticleClass(
            variables=[
                Variable("vara", dtype=np.float32, initial=1.0),
                Variable("varb", dtype=np.float32, initial=2.0),
            ]
        ),
        Particle,
    ],
)
@pytest.mark.parametrize("nparticles", [5, 10])
def test_create_particle_data(particle, nparticles):
    time_interval = TimeInterval(TIME[0], TIME[-1])
    ngrids = 4
    data = create_particle_data(pclass=particle, nparticles=nparticles, ngrids=ngrids, time_interval=time_interval)

    assert isinstance(data, dict)
    assert len(data) == len(particle.variables) + 1  # ei variable is separate

    variables = {var.name: var for var in particle.variables}

    for variable_name in variables.keys():
        variable = variables[variable_name]
        variable_array = data[variable_name]

        assert variable_array.shape[0] == nparticles

        dtype = variable.dtype
        if dtype is _SAME_AS_FIELDSET_TIME_INTERVAL.VALUE:
            dtype = type(time_interval.left)

        assert variable_array.dtype == dtype
