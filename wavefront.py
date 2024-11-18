import numpy as np


class Wavefront(object):
    """Represents a generic wavefront object."""

    def __init__(self, wavelength: float):
        self.wavelength = wavelength

    def generate_field(self, grid_size: int, extent: float, z: float) -> np.ndarray:
        """
        polymorphism B)
        """
        raise NotImplementedError("Subclasses must implement this method.")


class GaussianBeam(Wavefront):
    """
    Represents a Gaussian beam with specified parameters.
    """

    def __init__(self, wavelength: float, E0: float, w0: float):
        """
        Initialize a Gaussian beam with given parameters.

        Args:
            wavelength (float): Wavelength of the laser beam (in meters).
            E0 (float): Peak electric field amplitude at the beam waist (in volts per meter).
            w0 (float): The beam waist radius (the minimum beam size, in meters).
        """
        super().__init__(wavelength)
        self.E0 = E0
        self.w0 = w0

    def generate_field(self, z: float, grid_size: int, extent: float) -> np.ndarray:
        """
        Calculate the Gaussian beam field distribution at a distance z.

        Args:
            z (float): The axial distance from the beam's waist to the observation plane (in meters).
            grid_size (int): The size of the 2D grid (number of points in x and y directions).
            extent (float): The extent of the spatial grid (in meters).

        Returns:
            np.ndarray: A 2D array representing the electric field distribution at distance z.
        """
        x = np.linspace(-extent / 2, extent / 2, grid_size)
        y = np.linspace(-extent / 2, extent / 2, grid_size)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)

        k = 2 * np.pi / self.wavelength

        # Compute beam properties at z
        z_R = np.pi * self.w0**2 / self.wavelength
        wz = self.w0 * np.sqrt(1 + (z / z_R) ** 2)
        Rz = z * (1 + (z_R / z) ** 2) if z != 0 else np.inf
        psiz = np.arctan(z / z_R)

        E_z = (
            self.E0
            * (self.w0 / wz)
            * np.exp(-(r**2) / wz**2)
            * np.exp(-1j * k * r**2 / (2 * Rz))
            * np.exp(-1j * psiz)
        )

        return E_z


class PointSource(Wavefront):
    """
    Represents a point source generating a spherical wavefront.
    """

    def __init__(self, wavelength: float, E0: float):
        """
        Initialize a point source with given parameters.

        Args:
            wavelength (float): Wavelength of the light (in meters).
            E0 (float): Peak electric field amplitude at the point source (in volts per meter).
        """
        super().__init__(wavelength)
        self.E0 = E0

    def generate_field(self, grid_size: int, extent: float, z: float) -> np.ndarray:
        """
        Generate the electric field distribution for the point source (spherical wave).

        Args:
            grid_size (int): The size of the 2D grid (number of points in x and y directions).
            extent (float): The extent of the spatial grid (in meters).
            z (float): The axial distance from the point source to the observation plane (in meters).

        Returns:
            np.ndarray: A 2D array representing the electric field distribution of the spherical wave.
        """
        k = 2 * np.pi / self.wavelength

        # def spatial grid
        x = np.linspace(-extent / 2, extent / 2, grid_size)
        y = np.linspace(-extent / 2, extent / 2, grid_size)
        X, Y = np.meshgrid(x, y)

        r = np.sqrt(X**2 + Y**2 + z**2)

        E_r = (
            self.E0 * np.exp(1j * k * r) / r
        )  # amplitude decays as 1/r with phase exp(i * k * r)

        return E_r
