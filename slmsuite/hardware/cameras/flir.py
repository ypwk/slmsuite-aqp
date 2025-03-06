import warnings

from .camera import Camera

try:
    import PySpin
except ImportError:
    PySpin = None
    warnings.warn("PySpin not installed. Install to use FLIR cameras.")


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

    ### Initialization and termination ###

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
        # Note: using FLIR.sdk instead of PySpin.sdk
        camera_list = FLIR.sdk.GetCameras()
        if verbose:
            print(f"found {camera_list.GetSize()} cameras.")
            print("success")

        if verbose:
            print("Camera initializing... ", end="")
        if serial == "":
            self.cam = camera_list.GetByIndex(0)
        else:
            self.cam = camera_list.GetBySerial(serial)
        self.cam.Init()

        # Initialize the base Camera class with sensor dimensions and bitdepth.
        super().__init__(
            (self.cam.SensorWidth.get(), self.cam.SensorHeight.get()),
            bitdepth=int(self.cam.PixelSize.get()),
            pitch_um=pitch_um,
            name=serial,
            **kwargs,
        )

        # Optionally disable auto exposure by setting the node (if available)
        try:
            self.cam.ExposureAuto.set(PySpin.ExposureAuto_Off)
        except Exception as e:
            warnings.warn("Could not disable auto exposure: " + str(e))

        # Begin image acquisition so _get_image_hw works later.
        self.cam.BeginAcquisition()

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

    ### Property Configuration ###

    def get_exposure(self):
        """Get the current exposure time in seconds."""
        # Assume the camera returns exposure in microseconds.
        exposure_us = self.cam.ExposureTime.get()
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

        self.cam.ExposureTime.set(exposure_us)

    def set_woi(self, window=None):
        """Set the window of interest (WOI) for the camera.

        Parameters
        ----------
        window : tuple or None
            Tuple in the form (x_offset, y_offset, width, height). If None, resets to full sensor.
        """
        sensor_width = self.cam.SensorWidth.get()
        sensor_height = self.cam.SensorHeight.get()

        if window is None:
            window = (0, 0, sensor_width, sensor_height)
        else:
            if len(window) != 4:
                raise ValueError(
                    "Window must be a tuple of (x_offset, y_offset, width, height)."
                )

        x_offset, y_offset, width, height = window

        # Check that the window is within sensor bounds.
        if (
            x_offset < 0
            or y_offset < 0
            or x_offset + width > sensor_width
            or y_offset + height > sensor_height
        ):
            raise ValueError("Window dimensions are out of sensor bounds.")

        self.cam.OffsetX.set(x_offset)
        self.cam.OffsetY.set(y_offset)
        self.cam.Width.set(width)
        self.cam.Height.set(height)

    def _get_image_hw(self, blocking=True):
        """
        Get an image from the camera hardware.

        Parameters
        ----------
        blocking : bool
            Whether to wait for the camera to return a frame, blocking other acquisition.
        """
        timeout = (
            PySpin.EVENT_TIMEOUT_INFINITE if blocking else PySpin.EVENT_TIMEOUT_NONE
        )
        frame = self.cam.GetNextImage(timeout)
        image = frame.GetNDArray()
        frame.Release()  # Always release the frame to free memory.
        return image
