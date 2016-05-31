from parcels.field import Field
import numpy as np


def createSimpleGrid(x, y, time):
    field = np.zeros((time.size, x, y), dtype=np.float32)
    ltri = np.triu_indices(n=x, m=y)
    for t in time:
        temp = np.zeros((x, y), dtype=np.float32)
        temp[ltri] = 1
        field[t, :, :] = np.reshape(temp.T, np.shape(field[t, :, :]))

    return field

if __name__ == "__main__":
    x = 4
    y = 6
    time = np.linspace(0, 2, 3)
    field = Field("Test", data=createSimpleGrid(x, y, time), time=time, lon=np.linspace(0, x-1, x),
                  lat=np.linspace(-y/2, y/2-1, y))
    print("          ----- Raw Field Data -----")
    print(np.round(field.data[0, :, :], 0))
    grad_fields = field.gradient()
    # Use numpy gradient function for comparison, using fixed spacing of latitudinal cell distance
    r = 6.371e6
    deg2rd = np.pi / 180
    numpy_grad_fields = np.gradient(np.transpose(field.data[0, :, :]), (r * np.diff(field.lat) * deg2rd)[0])
    print("          ----- Field Gradient dx -----")
    print(grad_fields[0].data[0, :, :])
    print("          ----- Field Gradient dx (numpy, will be different) -----")
    print(np.array(np.transpose(numpy_grad_fields[0])))
    print("          ----- Field Gradient dy -----")
    print(grad_fields[1].data[0, :, :])
    print("          ----- Field Gradient dy (numpy, should be the same) -----")
    print(np.array(np.transpose(numpy_grad_fields[1])))

    # Time and space subsampled gradient field
    print("----- Subsampled field (middle section, first two timesteps) -----")
    grad_fields = field.gradient(timerange=[0, 1], lonrange=[1, 2], latrange=[-2, 1])
    print("          ----- Field Gradient dx -----")
    print(grad_fields[0].data)
    print("          ----- Field Gradient dy -----")
    print(grad_fields[1].data)
