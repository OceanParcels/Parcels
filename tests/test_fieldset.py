from datetime import timedelta

import cf_xarray  # noqa: F401
import cftime
import numpy as np
import pytest
import xarray as xr

from parcels._datasets.structured.circulation_models import datasets as datasets_circulation_models
from parcels._datasets.structured.generic import T as T_structured
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels.field import Field, VectorField
from parcels.fieldset import CalendarError, FieldSet, _datetime_to_msg
from parcels import XGrid
from tests import utils

ds = datasets_structured["ds_2d_left"]


@pytest.fixture
def fieldset() -> FieldSet:
    """Fixture to create a FieldSet object for testing."""
    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U (A grid)"], grid)
    V = Field("V", ds["V (A grid)"], grid)
    UV = VectorField("UV", U, V)

    return FieldSet(
        [U, V, UV],
    )


def test_fieldset_init_wrong_types():
    with pytest.raises(ValueError, match="Expected `field` to be a Field or VectorField object. Got .*"):
        FieldSet([1.0, 2.0, 3.0])


def test_fieldset_add_constant(fieldset):
    fieldset.add_constant("test_constant", 1.0)
    assert fieldset.test_constant == 1.0


def test_fieldset_add_constant_field(fieldset):
    fieldset.add_constant_field("test_constant_field", 1.0)

    # Get a point in the domain
    time = ds["time"].mean()
    depth = ds["depth"].mean()
    lat = ds["lat"].mean()
    lon = ds["lon"].mean()

    pytest.xfail(reason="Not yet implemented interpolation.")
    assert fieldset.test_constant_field[time, depth, lat, lon] == 1.0


def test_fieldset_add_field(fieldset):
    grid = XGrid.from_dataset(ds, mesh="flat")
    field = Field("test_field", ds["U (A grid)"], grid)
    fieldset.add_field(field)
    assert fieldset.test_field == field


def test_fieldset_add_field_wrong_type(fieldset):
    not_a_field = 1.0
    with pytest.raises(ValueError, match="Expected `field` to be a Field or VectorField object. Got .*"):
        fieldset.add_field(not_a_field, "test_field")


def test_fieldset_add_field_already_exists(fieldset):
    grid = XGrid.from_dataset(ds, mesh="flat")
    field = Field("test_field", ds["U (A grid)"], grid)
    fieldset.add_field(field, "test_field")
    with pytest.raises(ValueError, match="FieldSet already has a Field with name 'test_field'"):
        fieldset.add_field(field, "test_field")


def test_fieldset_gridset(fieldset):
    assert fieldset.fields["U"].grid in fieldset.gridset
    assert fieldset.fields["V"].grid in fieldset.gridset
    assert fieldset.fields["UV"].grid in fieldset.gridset
    assert len(fieldset.gridset) == 1

    fieldset.add_constant_field("constant_field", 1.0)
    assert len(fieldset.gridset) == 2


@pytest.mark.parametrize("ds", [pytest.param(ds, id=k) for k, ds in datasets_structured.items()])
def test_fieldset_from_structured_generic_datasets(ds):
    grid = XGrid.from_dataset(ds, mesh="flat")
    fields = []
    for var in ds.data_vars:
        fields.append(Field(var, ds[var], grid))

    fieldset = FieldSet(fields)

    assert len(fieldset.fields) == len(ds.data_vars)
    for field in fieldset.fields.values():
        utils.assert_valid_field_data(field.data, field.grid)

    assert len(fieldset.gridset) == 1


def test_fieldset_gridset_multiple_grids(): ...


def test_fieldset_time_interval():
    grid1 = XGrid.from_dataset(ds, mesh="flat")
    field1 = Field("field1", ds["U (A grid)"], grid1)

    ds2 = ds.copy()
    ds2["time"] = (ds2["time"].dims, ds2["time"].data + np.timedelta64(timedelta(days=1)), ds2["time"].attrs)
    grid2 = XGrid.from_dataset(ds2, mesh="flat")
    field2 = Field("field2", ds2["U (A grid)"], grid2)

    fieldset = FieldSet([field1, field2])
    fieldset.add_constant_field("constant_field", 1.0)

    assert fieldset.time_interval.left == np.datetime64("2000-01-02")
    assert fieldset.time_interval.right == np.datetime64("2001-01-01")


def test_fieldset_time_interval_constant_fields():
    fieldset = FieldSet([])
    fieldset.add_constant_field("constant_field", 1.0)
    fieldset.add_constant_field("constant_field2", 2.0)

    assert fieldset.time_interval is None


def test_fieldset_init_incompatible_calendars():
    ds1 = ds.copy()
    ds1["time"] = (
        ds1["time"].dims,
        xr.date_range("2000", "2001", T_structured, calendar="365_day", use_cftime=True),
        ds1["time"].attrs,
    )

    grid = XGrid.from_dataset(ds1, mesh="flat")
    U = Field("U", ds1["U (A grid)"], grid)
    V = Field("V", ds1["V (A grid)"], grid)
    UV = VectorField("UV", U, V)

    ds2 = ds.copy()
    ds2["time"] = (
        ds2["time"].dims,
        xr.date_range("2000", "2001", T_structured, calendar="360_day", use_cftime=True),
        ds2["time"].attrs,
    )
    grid2 = XGrid.from_dataset(ds2, mesh="flat")
    incompatible_calendar = Field("test", ds2["data_g"], grid2)

    with pytest.raises(CalendarError, match="Expected field '.*' to have calendar compatible with datetime object"):
        FieldSet([U, V, UV, incompatible_calendar])


def test_fieldset_add_field_incompatible_calendars(fieldset):
    ds_test = ds.copy()
    ds_test["time"] = (
        ds_test["time"].dims,
        xr.date_range("2000", "2001", T_structured, calendar="360_day", use_cftime=True),
        ds_test["time"].attrs,
    )
    grid = XGrid.from_dataset(ds_test, mesh="flat")
    field = Field("test_field", ds_test["data_g"], grid)

    with pytest.raises(CalendarError, match="Expected field '.*' to have calendar compatible with datetime object"):
        fieldset.add_field(field, "test_field")

    ds_test = ds.copy()
    ds_test["time"] = (
        ds_test["time"].dims,
        np.linspace(0, 100, T_structured, dtype="timedelta64[s]"),
        ds_test["time"].attrs,
    )
    grid = XGrid.from_dataset(ds_test, mesh="flat")
    field = Field("test_field", ds_test["data_g"], grid)

    with pytest.raises(CalendarError, match="Expected field '.*' to have calendar compatible with datetime object"):
        fieldset.add_field(field, "test_field")


@pytest.mark.parametrize(
    "input_, expected",
    [
        (cftime.DatetimeNoLeap(2000, 1, 1), "<class 'cftime._cftime.DatetimeNoLeap'> with cftime calendar noleap'"),
        (cftime.Datetime360Day(2000, 1, 1), "<class 'cftime._cftime.Datetime360Day'> with cftime calendar 360_day'"),
        (cftime.DatetimeJulian(2000, 1, 1), "<class 'cftime._cftime.DatetimeJulian'> with cftime calendar julian'"),
        (
            cftime.DatetimeGregorian(2000, 1, 1),
            "<class 'cftime._cftime.DatetimeGregorian'> with cftime calendar standard'",
        ),
        (np.datetime64("2000-01-01"), "<class 'numpy.datetime64'>"),
        (cftime.datetime(2000, 1, 1), "<class 'cftime._cftime.datetime'> with cftime calendar standard'"),
    ],
)
def test_datetime_to_msg(input_, expected):
    assert _datetime_to_msg(input_) == expected


def test_fieldset_samegrids_UV():
    """Test that if a simple fieldset with U and V is created, that only one grid object is defined."""
    ...


def test_fieldset_grid_deduplication():
    """Tests that for a full fieldset that the number of grid objects is as expected
    (sharing of grid objects so that the particle location is not duplicated).

    When grid deduplication is actually implemented, this might need to be refactored
    into multiple tests (/more might be needed).
    """
    ...


def test_fieldset_add_field_after_pset():
    # ? Should it be allowed to add fields (normal or vector) after a ParticleSet has been initialized?
    ...


_COPERNICUS_DATASETS = [
    datasets_circulation_models["ds_copernicusmarine"],
    datasets_circulation_models["ds_copernicusmarine_waves"],
]


@pytest.mark.parametrize("ds", _COPERNICUS_DATASETS)
def test_fieldset_from_copernicusmarine(ds, caplog):
    fieldset = FieldSet.from_copernicusmarine(ds)
    assert "U" in fieldset.fields
    assert "V" in fieldset.fields
    assert "UV" in fieldset.fields
    assert "renamed it to 'U'" in caplog.text
    assert "renamed it to 'V'" in caplog.text


def test_fieldset_from_copernicusmarine_no_currents(caplog):
    ds = datasets_circulation_models["ds_copernicusmarine"].cf.drop_vars(
        ["eastward_sea_water_velocity", "northward_sea_water_velocity"]
    )
    fieldset = FieldSet.from_copernicusmarine(ds)
    assert "U" not in fieldset.fields
    assert "V" not in fieldset.fields
    assert "UV" not in fieldset.fields
    assert caplog.text == ""


@pytest.mark.parametrize("ds", _COPERNICUS_DATASETS)
def test_fieldset_from_copernicusmarine_no_logs(ds, caplog):
    ds = ds.copy()
    zeros = xr.zeros_like(list(ds.data_vars.values())[0])
    ds["U"] = zeros
    ds["V"] = zeros

    fieldset = FieldSet.from_copernicusmarine(ds)
    assert "U" in fieldset.fields
    assert "V" in fieldset.fields
    assert "UV" in fieldset.fields
    assert caplog.text == ""


def test_fieldset_from_copernicusmarine_with_W(caplog):
    ds = datasets_circulation_models["ds_copernicusmarine"]
    ds = ds.copy()
    ds["wo"] = ds["uo"]
    ds["wo"].attrs["standard_name"] = "vertical_sea_water_velocity"

    fieldset = FieldSet.from_copernicusmarine(ds)
    assert "U" in fieldset.fields
    assert "V" in fieldset.fields
    assert "W" in fieldset.fields
    assert "UV" not in fieldset.fields
    assert "UVW" in fieldset.fields
    assert "renamed it to 'W'" in caplog.text
