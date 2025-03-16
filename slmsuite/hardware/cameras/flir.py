import warnings

from .camera import Camera

try:
    import PySpin
    print("imported pyspin")
except ImportError:
    PySpin = None
    warnings.warn("PySpin not installed. Install to use FLIR cameras.")

import time
import numpy as np


class FLIR(Camera):
    """
    FLIR camera.

    Attributes
    ----------
    sdk : PySpin.System
        Spinnaker SDK.
    cam : PySpin.Camera
        Object to talk with the desired camera.
    """

    sdk = None

    # Initialization and termination

    def __init__(self, serial="", pitch_um=None, verbose=True, **kwargs):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        serial : str
            Serial number of the camera to open.
        pitch_um : (float, float) OR None
            Fill in extra information about the pixel pitch in ``(dx_um, dy_um)`` form
            to use additional calibrations.
        verbose : bool
            Whether or not to log.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        if PySpin is None:
            raise ImportError("PySpin not installed. Install to use FLIR cameras.")

        if FLIR.sdk is None:
            if verbose:
                print("PySpin initializing... ", end="")
            FLIR.sdk = PySpin.System.GetInstance()

            if verbose:
                print("success")

        if verbose:
            print("Looking for cameras... ", end="")

        camera_list = FLIR.sdk.GetCameras()
        if verbose:
            print("success")

        if verbose:
            print("Camera initializing... ", end="")
        if serial == "":
            self.cam = camera_list.GetByIndex(0)
        else:
            self.cam = camera_list.GetBySerial(serial)
        self.cam.Init()

        # Initialize the base Camera class with sensor dimensions and bitdepth.

        # Optionally disable auto exposure by setting the node (if available)
        try:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        except Exception as e:
            warnings.warn("Could not disable auto exposure: " + str(e))

        if not self.cam.IsStreaming():
            self.cam.BeginAcquisition()
        
        super().__init__(
            (self.cam.SensorWidth, self.cam.SensorHeight),
            bitdepth=int(self.cam.PixelSize()),
            pitch_um=pitch_um,
            name=serial,
            **kwargs,
        )

    def close(self, close_sdk=True):
        """Cleanly end acquisition and deinitialize the camera."""
        try:
            self.cam.EndAcquisition()
        except Exception as e:
            warnings.warn("Error ending acquisition: " + str(e))
        self.cam.DeInit()
        del self.cam

        if close_sdk and FLIR.sdk is not None:
            FLIR.sdk.ReleaseInstance()
            FLIR.sdk = None

    # Property Configuration

    def get_exposure(self):
        """Get the current exposure time in seconds."""
        # Assume the camera returns exposure in microseconds.
        exposure_us = self.cam.ExposureTime()
        return exposure_us / 1e6

    def set_exposure(self, exposure_s):
        """Set the camera exposure time in seconds."""
        # Ensure auto exposure is off
        self.cam.ExposureAuto.set(PySpin.ExposureAuto_Off)
        exposure_us = exposure_s * 1e6

        # Optionally check camera limits, if available.
        try:
            exposure_min = self.cam.ExposureTime.get_min()
            exposure_max = self.cam.ExposureTime.get_max()
            if exposure_us < exposure_min or exposure_us > exposure_max:
                warnings.warn(
                    f"Requested exposure {exposure_s}s is out of bounds "
                    f"({exposure_min/1e6:.3f}s - {exposure_max/1e6:.3f}s). Clipping to valid range."
                )
                exposure_us = max(min(exposure_us, exposure_max), exposure_min)
        except Exception:
            # If limits are not available, proceed without checking.
            pass

        self.cam.ExposureTime.SetValue(exposure_us)
        time.sleep(0.5)  # Wait for exposure to settle.

    def set_woi(self, window=None):
        """See :meth:`.Camera.set_woi`."""
        return

    def _get_image_hw(self, timeout_s = 5):
        """
        Fetches a single image from the camera hardware with a specified timeout.

        Parameters
        ----------
        timeout_s : float
            The time in seconds to wait for each frame to be fetched.
            Use a negative value for blocking (infinite wait) and 0 for non-blocking mode.

        Returns
        -------
        image : numpy.ndarray
            The captured image as a NumPy array (dtype: uint8).
        """
        # Determine the timeout based on timeout_s
        if timeout_s < 0:
            timeout = PySpin.EVENT_TIMEOUT_INFINITE
        elif timeout_s == 0:
            timeout = PySpin.EVENT_TIMEOUT_NONE
        else:
            # Convert seconds to milliseconds (PySpin typically expects ms)
            timeout = int(timeout_s * 1000)

        frame = self.cam.GetNextImage(timeout)

        # Check that the frame is valid
        if not frame.IsValid():
            raise RuntimeError("Failed to acquire a valid image frame.")

        width = frame.GetWidth()
        height = frame.GetHeight()
        # Convert the frame data into a NumPy array and reshape it based on image dimensions.
        image = np.array(frame.GetData(), dtype=np.uint8).reshape((height, width))

        frame.Release()  # Always release the frame to free memory.

        return image
