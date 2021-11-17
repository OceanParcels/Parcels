from numba.core.typing.asnumbatype import as_numba_type
from .field import NumbaField
from parcels.numba.field.vector_field_2d import NumbaVectorField2D
from parcels.numba.field.vector_field_3d import NumbaVectorField3D
from numba.experimental import jitclass
from parcels.field import Field, VectorField


def _base_fieldset_spec():
    return [
        ("U", as_numba_type(NumbaField)),
        ("V", as_numba_type(NumbaField)),
        ("W", as_numba_type(NumbaField)),
        ("UV", as_numba_type(NumbaVectorField2D)),
        ("UVW", as_numba_type(NumbaVectorField3D))
    ]


class NumbaFieldSet():
    @classmethod
    def create(cls, U, V, W=None, fields={}):
        field_class = U.numba_class
        field_class_2d = NumbaVectorField2D._class(U)
        field_class_3d = NumbaVectorField3D._class(U)

        spec = [
            ("U", as_numba_type(field_class)),
            ("V", as_numba_type(field_class)),
            ("UV", as_numba_type(field_class_2d)),
        ]
        if W is not None:
            spec.append(("W", as_numba_type(field_class)))
            spec.append(("UVW", as_numba_type(field_class_3d)))

        for name, field in fields.items():
            if isinstance(field, Field):
                spec.append((name, as_numba_type(field.numba_class)))
            elif isinstance(field, VectorField):
                spec.append((name, as_numba_type(field.numba_class)))
            else:
                raise TypeError(f"'{name}' Field should be scalar or 2D vector field ")

        new_class = jitclass(BaseNumbaFieldSet, spec=spec)
        numba_fieldset = new_class()
        numba_fieldset.U = U.numba_field
        numba_fieldset.V = V.numba_field
        if W is not None:
            numba_fieldset.W = W.numba_field
            numba_fieldset.UVW = field_class_3d("UVW", U.numba_field, V.numba_field, W.numba_field)
        numba_fieldset.UV = field_class_2d("UV", U.numba_field, V.numba_field)
        for name, field in fields.items():
            setattr(numba_fieldset, name, field.numba_field)
        return numba_fieldset

# 
# 
#     @staticmethod
#     def create_numba_fieldset(U, V, W=None, fields={}):
#         spec = _base_fieldset_spec()
#         for name, field in fields.items():
#             if isinstance(field, Field):
#                 spec.append((name, as_numba_type(NumbaField)))
#             elif isinstance(field, VectorField):
#                 spec.append((name, as_numba_type(NumbaVectorField2D)))
#             else:
#                 raise TypeError(f"'{name}' Field should be scalar or 2D vector field ")
#         numba_class = jitclass(BaseNumbaFieldSet, spec=spec)
#         if W is not None:
#             numba_fieldset = numba_class(U.numba_field, V.numba_field, W.numba_field)
#         else:
#             numba_fieldset = numba_class(U.numba_field, V.numba_field)
# 
#         for name, field in fields.items():
#             setattr(numba_fieldset, name, field.numba_field)
#         return numba_fieldset


class BaseNumbaFieldSet():
    def __init__(self):
        pass
#         self.U = U
#         self.V = V
#         self.UV = NumbaVectorField2D("UV", U, V)
#         if W is not None:
#             self.W = W
#             self.UVW = NumbaVectorField3D("UVW", U, V, W)
