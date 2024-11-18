from slmsuite.holography.algorithms import SpotHologram3D
from wavefront import GaussianBeam
import numpy as np

params = {
    "grid_size": 256,
    "extent": 500e-6,
    "wavelength": 809e-9,  # use levine wavelength
    "spacing": 10e-6,  # trap spacing
    "nx": 10,  # number of traps (x)
    "ny": 10,  # number of traps (y)
    "nz": 10,
    "E0": 1,
    "I0": 5.4e09,
    "w0": 343e-09,
    "z0": 0.0,
    "z1": 10,
    "driver": "aqp",  # aqp - kevin implement, slm - slmsuite
    "iterations": 100,  # gs iters
}


def calculate_trap_pitch(
    grid_size: int, x_traps: int, y_traps: int, extent: float, spacing: float
) -> float:
    if x_traps * spacing >= extent or y_traps * spacing >= extent:
        raise ValueError(
            "Too many traps for current extent! Either decrease the number of traps or increase the extent."
        )
    return spacing * grid_size / extent


gaussian_beam = GaussianBeam(
    wavelength=params["wavelength"], E0=params["E0"], w0=params["w0"]
)
gaussian_field = gaussian_beam.generate_field(
    grid_size=params["grid_size"], extent=params["extent"], z=params["z1"]
)

gaussian_amplitude = np.abs(gaussian_field)
with np.errstate(divide="ignore", invalid="ignore"):
    gaussian_phase = np.angle(gaussian_field)
    gaussian_phase = np.where(gaussian_amplitude == 0, 0, gaussian_phase)

trap_pitch = calculate_trap_pitch(
    params["grid_size"], params["nx"], params["ny"], params["extent"], params["spacing"]
)

# Instead of picking a few points, make a rectangular grid in the knm basis
array_holo = SpotHologram3D.make_rectangular_array(
    (params["grid_size"], params["grid_size"], params["grid_size"]),
    array_shape=(params["nx"], params["ny"], params["nz"]),
    array_pitch=(trap_pitch, trap_pitch, trap_pitch),
    basis="knm",
)

array_holo.reset_phase(gaussian_phase)

array_holo.optimize(
    method="WGS-Kim", maxiter=20, mraf_factor=0.5, stat_groups=["computational_spot"]
)
array_holo.plot_stats()
