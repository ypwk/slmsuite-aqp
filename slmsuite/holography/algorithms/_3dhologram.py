from slmsuite.holography.algorithms._header import *
from slmsuite.holography.algorithms._stats import _HologramStats


"""
This file contains code that reproduces what is done in Barredo et al. for 3D optical trap arrays. 
"""


class Hologram3D(_HologramStats):
    r"""
    Phase retrieval methods applied to holography (DFT-based).
    See :meth:`.optimize()` to learn about the methods implemented for hologram optimization.

    Tip
    ~~~
    The Fourier domain (``"kxy"``) of an SLM with shape :attr:`slm_shape` also has the shape
    :attr:`slm_shape` under discrete Fourier transform. The extents of this
    domain correspond to the edges of the farfield determined by physical constants
    as :math:`\pm\frac{\lambda}{2\Delta x}` radians, where :math:`\Delta x`
    is the SLM pixel pitch. This means that resolution of the farfield
    :math:`\pm\frac{\lambda}{2N_x\Delta x}` radians can be quite poor with small
    resolution :math:`N_x`. The solution is to zero-pad the SLM nearfield
    --- artificially increasing the width :math:`N_x` and height
    :math:`N_y` while the extent of the non-zero nearfield data remains the same ---
    and thus enhance the resolution of the farfield.
    In practice, padding is accomplished by passing a :attr:`shape` or
    :attr:`target` of appropriate shape (see constructor :meth:`.__init__()` and subclasses),
    potentially with the aid of the static helper function :meth:`.get_padded_shape()`.

    Note
    ~~~~
    :attr:`target`, :attr:`weights`, :attr:`phase_ff`, and :attr:`amp_ff` are all
    matrices of shape :attr:`shape`. To save memory, the matrices :attr:`phase` and :attr:`amp`
    are stored with the (smaller, but not strictly smaller) shape :attr:`slm_shape`.
    Also to save memory, :attr:`phase_ff` and :attr:`amp_ff` are set to ``None`` on construction,
    and only initialized if they need to be used. Any code additions should check for ``None``.

    Tip
    ~~~
    Due to imperfect SLM diffraction efficiency, undiffracted light will
    be present at the center of the :attr:`target`.
    This is called the zeroth order diffraction peak.
    To avoid this peak, consider shifting
    the data contained in :attr:`target` away from the center.

    Tip
    ~~~
    For mixed region amplitude freedom (MRAF) capabilities, set the part of the
    :attr:`target` desired as a 'noise region' to ``nan``.
    See :meth:`optimize()` for more details.
    :class:`SpotHologram` has spot-specific methods for generated noise region pattern.

    Caution
    ~~~~~~~
    By default, arguments passed to the constructor (:attr:`phase`, :attr:`amp`, ... )
    are stored directly as attributes **without copying**, when possible. These will be
    modified in place. However, :mod:`numpy` arrays passed to :mod:`cupy` will naturally
    be copied onto the GPU and arrays of incorrect :attr:`dtype` will likewise be copied
    and casted. This lack of copying is desired in many cases, such that external routines
    access the same data, but the user can choose to pass copied arrays if this behavior is undesired.

    Attributes
    ----------
    slm_shape : (int, int)
        The shape of the **nearfield** device producing the hologram in the **farfield**
        in :mod:`numpy` ``(h, w)`` form. This is important to record because
        certain optimizations and calibrations depend on it. If multiple of :attr:`slm_shape`,
        :attr:`phase`, or :attr:`amp` are not ``None`` in the constructor, the shapes must agree.
        If all are ``None``, then the shape of the :attr:`target` is used instead
        (:attr:`slm_shape` == :attr:`shape`).
    shape : (int, int)
        The shape of the computational space in the **nearfield** and **farfield**
        in :mod:`numpy` ``(h, w)`` form.
        Corresponds to the the ``"knm"`` basis in the **farfield**.
        This often differs from :attr:`slm_shape` due to padding of the **nearfield**.
    phase : numpy.ndarray OR cupy.ndarray
        **nearfield** phase pattern to optimize.
        Initialized to with :meth:`random.default_rng().uniform()` by default (``None``).
        This is of shape :attr:`slm_shape`
        and (upon copying to :attr:`nearfield` during optimization)
        padded to shape :attr:`shape`.
    amp : numpy.ndarray OR cupy.ndarray
        **nearfield** source amplitude pattern (i.e. image-space constraints).
        Uniform illumination is assumed by default (``None``).
        This is of shape :attr:`slm_shape`
        and (upon copying to :attr:`nearfield` during optimization)
        padded to shape :attr:`shape`.
    nearfield : numpy.ndarray OR cupy.ndarray
        Helper variable to encode the data in
        :attr:`phase` and :attr:`amp` as a single complex matrix.
        This is of shape :attr:`shape`.
    target : numpy.ndarray OR cupy.ndarray
        Desired **farfield** amplitude in the ``"knm"`` basis. The goal of optimization.
        This is of shape :attr:`shape` in DFT-based algorithms, but differs for
        :class:`CompressedSpotHologram`.
    weights : numpy.ndarray OR cupy.ndarray
        The mutable **farfield** amplitude in the ``"knm"`` basis used in GS.
        Starts as :attr:`target` but may be modified by weighted feedback in WGS.
        This is of the same shape as :attr:`target`.
    farfield : numpy.ndarray OR cupy.ndarray
        Helper variable to encode the data in the farfield as a single complex matrix.
        This is of the same shape as :attr:`target`.
    phase_ff : numpy.ndarray OR cupy.ndarray
        Algorithm-constrained **farfield** phase in the ``"knm"`` basis.
        Used as a helper variable for optimization.
        Stored for computational algorithms which desire to fix the phase in the farfield
        (see :meth:`~slmsuite.holography.algorithms.Hologram.optimize_gs`).
        This is of the same shape as :attr:`target`.
    amp_ff : numpy.ndarray OR cupy.ndarray OR None
        **Farfield** amplitude in the ``"knm"`` basis.
        Used for comparing this, the computational result, with the :attr:`target`.
        This is of the same shape as :attr:`target`.
    dtype : type
        Datatype for real arrays.
        The default is ``float32``.
    dtype_complex : type
        Datatype for complex arrays. The complex numbers follow :mod:`numpy`
        `type promotion <https://numpy.org/doc/stable/reference/routines.fft.html#type-promotion>`_.
        Complex datatypes are derived from :attr:`dtype`:

        - ``float32`` -> ``complex64`` (the default :attr:`dtype`)
        - ``float64`` -> ``complex128``

        ``float16`` is *not* recommended for :attr:`dtype` because ``complex32`` is not
        implemented by :mod:`numpy`.
    propagation_kernel : numpy.ndarray OR cupy.ndarray OR None
        Allows the user to target holography at different depths or aberration spaces.
        This is also applied for
        :class:`~slmsuite.holography.algorithms.FeedbackHologram`
        and subclasses to `~slmsuite.holography.algorithms.FeedbackHologram.measure()`
        the hologram at the desired plane.
        If ``None``, this feature is not used and no depth or aberration
        transformation is applied.
    iter : int
        Tracks the current iteration number.
    flags : dict
        Helper flags to store custom persistent variables for optimization.
        These flags are generally changed by passing as a ``kwarg`` to
        :meth:`~slmsuite.holography.algorithms.Hologram.optimize()`.
        Contains the following keys:

         - ``"method"`` : ``str``
            Stores the method used for optimization.
            See :meth:`~slmsuite.holography.algorithms.Hologram.optimize()`.
         - ``"fixed_phase"`` : ``bool``
            Fixes the farfield phase as mandated by certain weighted algorithms
            (see :meth:`~slmsuite.holography.algorithms.Hologram.optimize_gs()`).
         - ``"feedback"`` : ``str``
            Stores the values passed to
            :meth:`~slmsuite.holography.algorithms.Hologram.optimize()`.
         - ``"stat_groups"`` : ``list of str``
            Stores the values passed to
            :meth:`~slmsuite.holography.algorithms.Hologram.optimize()`.
         - ``"raw_stats"`` : ``bool``
            Whether to store raw stats: the raw image and feedback data for each
            iteration. Note that this can be a good amount of data.
         - ``"blur_ij"`` : ``float``
            See :meth:`~slmsuite.holography.algorithms.FeedbackHologram.ijcam_to_knmslm()`.
         - Other user-defined flags.

    stats : dict
        Dictionary of useful statistics. data is stored in lists, with indices corresponding
        to each iteration. Contains:

         - ``"methods"`` : ``list of str``
            Method used for each iteration.
         - ``"flags"`` : ``dict of lists``
            Each key corresponds to a flag that was used at least once. If it is ``np.nan``
            on a given iteration, then it was undefined at that point (update functions
            keep track of all this).
         - ``"stats"`` : ``dict of dicts of lists``
            Same format as ``"flags"``, except with another layer of hierarchy corresponding
            to the source (group) of the given stats. This is to differentiate standard deviations
            computed computationally and experimentally.

        See :meth:`._update_stats()` and :meth:`.plot_stats()`.
    """

    def __init__(
        self,
        target,
        amp=None,
        phase=None,
        slm_shape=None,
        dtype=np.float32,
        propagation_kernel=None,
        **kwargs,
    ):
        r"""
        Initialize datastructures for optimization.
        When :mod:`cupy` is enabled, arrays are initialized on the GPU as :mod:`cupy` arrays:
        take care to use class methods to access these parameters instead of editing
        them directly, as :mod:`cupy` arrays behave slightly differently than numpy arrays in
        some cases.

        Parameters additional to class attributes are described below:

        Parameters
        ----------
        target : numpy.ndarray OR cupy.ndarray OR (int, int) OR None
            Target to optimize to.
            The user can also pass a shape in :mod:`numpy` ``(h, w)`` form,
            and this constructor will create an empty target of all zeros.
            :meth:`.get_padded_shape()` can be of particular help for calculating the
            shape that will produce desired results (in terms of precision, etc).
            ``None`` is used internally.
        amp : array_like OR None
            The nearfield amplitude. See :attr:`amp`. Of shape :attr:`slm_shape`.
        phase : array_like OR None
            The nearfield initial phase.
            See :attr:`phase`. :attr:`phase` should only be passed if the user wants to
            precondition the optimization. Of shape :attr:`slm_shape`.
        slm_shape : (int, int) OR slmsuite.hardware.FourierSLM OR slmsuite.hardware.slms.SLM OR None
            The shape of the nearfield of the SLM in :mod:`numpy` `(h, w)` form.
            Optionally, as a quality of life feature, the user can pass a
            :class:`~slmsuite.hardware.FourierSLM` or
            :class:`~slmsuite.hardware.slms.SLM` instead,
            and ``slm_shape`` (and ``amp`` if it is ``None``) are populated from this.
            If ``None``, tries to use the shape of ``amp`` or ``phase``, but if these
            are not present, defaults to :attr:`shape` (which is usually determined by ``target``).
        dtype : type
            See :attr:`dtype`; the type to use for stored arrays. The user should choose
            this as a tradeoff between precision, memory size, and compute time.
        propagation_kernel : array_like OR None
            Primarily used for targeting holography at a different depth plane, encoded
            by a focusing kernel. :class:`MultiplaneHologram`, targeting several depth planes,
            must make use of this parameter to 'bake' the information for each plane
            into the composite hologram.
            A more advanced use of this feature is to
            target different positions in aberration-space, i.e. have a unique
            wavefront calibration baked into the hologram for each plane.

            The kernel must be of shape :attr:`slm_shape`.
            If ``None``, this feature is unused (the kernel is an ideal DFT).

            Tip
            ~~~
            The unit conversions necessary to convert a depth into a Zernike focusing
            parameter are stored in a ``cameraslm``, and can be accessed via
            :meth:`~slmsuite.holography.toolbox.convert_vector`.

            .. highlight:: python
            .. code-block:: python

                # Convert a floating point Z depth from a desired units to Zernike units.
                depth_zernike = convert_vector(
                    (0, 0, depth),
                    from_units="{depth_units}",
                    to_units="zernike",
                    hardware=cameraslm,
                )

                # The ANSI Zernike indices (2,1,4) [x,y,z]
                # are automatically assumed from the 3-vector.
                propagation_kernel = toolbox.phase.zernike_sum(
                    grid=cameraslm,
                    weights=depth_zernike,
                )

            Note
            ~~~~
            Is ignored for :class:`~slmsuite.holography.algorithms.CompressedSpotHologram`.
        **kwargs
            Passed to :attr:`flags`.
        """
        # 1) Determine the shape of the SLM. We have three sources of this shape, which are
        # optional to pass, but must be self-consistent if passed:
        #
        #  a) The shape of the nearfield amplitude
        #  b) The shape of the seed nearfield phase
        #  c) slm_shape itself (which is set to the shape of a passed SLM, if given).
        #
        # If no parameter is passed for a given shape, the shape is set to (nan, nan) to
        # prepare for a vote (next section).

        # Option a
        if amp is None:
            amp_shape = (np.nan, np.nan)
        else:
            amp_shape = amp.shape

        # Option b
        if phase is None:
            phase_shape = (np.nan, np.nan)
        else:
            phase_shape = phase.shape

        # Option c
        if slm_shape is None:
            slm_shape = (np.nan, np.nan)
        else:
            try:  # Check if slm_shape is a CameraSLM.
                if amp is None:
                    amp = slm_shape.slm._get_source_amplitude()
                    amp_shape = amp.shape
                slm_shape = slm_shape.slm.shape
            except:
                try:  # Check if slm_shape is an SLM
                    if amp is None:
                        amp = slm_shape._get_source_amplitude()
                        amp_shape = amp.shape
                    slm_shape = slm_shape.shape

                except:  # (int, int) case
                    pass

            if len(slm_shape) != 2:
                slm_shape = (np.nan, np.nan)

        # 1) [cont] We now have a few options for what the shape of the SLM could be.
        # Parse these to validate consistency.
        stack = np.vstack((amp_shape, phase_shape, slm_shape))
        if np.all(np.isnan(stack)):
            self.slm_shape = None
        else:
            self.slm_shape = np.rint(np.nanmean(stack, axis=0)).astype(int)

            if amp is not None:
                if not np.all(self.slm_shape == np.array(amp_shape)):
                    raise ValueError(
                        "The shape of amplitude (via `amp` or SLM) is not equal to the "
                        "shapes of the provided initial phase (`phase`) or SLM (via `target` or `slm_shape`)"
                    )
            if phase is not None:
                if not np.all(self.slm_shape == np.array(phase_shape)):
                    raise ValueError(
                        "The shape of the initial phase (`phase`) is not equal to the "
                        "shapes of the provided amplitude (via `amp` or SLM) or SLM (via `target` or `slm_shape`)"
                    )
            if slm_shape is not None:
                if not np.all(self.slm_shape == np.array(slm_shape)):
                    raise ValueError(
                        "The shape of SLM (via `target` or `slm_shape`) is not equal to the "
                        "shapes of the provided initial phase (`phase`) or amplitude (via `amp` or SLM)"
                    )

            self.slm_shape = tuple(self.slm_shape)

        # 1.5) Parse target and create shape.
        if target is None:  # Multi or Compressed Hologram.
            if self.slm_shape is None:
                raise ValueError("SLM shape must be provided through cameraslm=")

            self.shape = self.slm_shape

            # Don't initialize memory for Multi
            if target is None:
                target = []
        else:  # Other cases
            if len(target) == 3:  # (int, int, int) was passed.
                self.shape = target[
                    :2
                ]  # slm should match first two dimensions of target
                self.target_shape = target
                target = None
            elif len(np.shape(target)) == 3:  # array_like (true target) passed.
                self.shape = np.shape(target)[:2]
                self.target_shape = np.shape(target)
            else:
                raise ValueError(f"Unexpected target {target}.")

            # Warn the user about powers of two if not multiplane hologram.
            if any(
                np.log2(self.shape) != np.round(np.log2(self.shape))
            ) and not hasattr(self, "holograms"):
                warnings.warn(
                    f"Hologram target shape {self.shape} is not a power of 2; consider using "
                    ".get_padded_shape() to pad to powers of 2 and speed up "
                    "FFT computation. While some FFT solvers support other prime powers "
                    "(3, 5, 7, ...), literature suggests that GPU support is best for powers of 2."
                )

        if self.slm_shape is None:
            self.slm_shape = self.shape

        # 2) Initialize variables.
        # Save the data type.
        if dtype(0).nbytes == 4:
            self.dtype = np.float32
            self.dtype_complex = np.complex64
        elif dtype(0).nbytes == 8:
            self.dtype = np.float64
            self.dtype_complex = np.complex128
        else:
            raise ValueError(f"Data type {dtype} not supported.")

        # Initialize and normalize nearfield amplitude.
        if amp is None:  # Uniform amplitude by default (scalar).
            self.amp = 1 / np.sqrt(np.prod(self.slm_shape))
        else:  # Otherwise, initialize and normalize.
            self.amp = cp.array(
                amp,
                dtype=self.dtype,
                copy=(False if np.__version__[0] == "1" else None),
            )
            self.amp *= 1 / Hologram._norm(self.amp)

        # Check propagation_kernel.
        if propagation_kernel is None:
            self.propagation_kernel = None
        elif isinstance(propagation_kernel, REAL_TYPES):
            self.propagation_kernel
        else:
            self.propagation_kernel = cp.array(
                propagation_kernel,
                dtype=self.dtype,
                copy=(False if np.__version__[0] == "1" else None),
            )
            if self.propagation_kernel.shape != self.slm_shape:
                raise ValueError(
                    "Expected the propagation kernel to be the same shape as the SLM."
                )

        # Initialize flags.
        self.flags = kwargs

        # Initialize target. reset() will handle weights.
        self._set_target(target, reset_weights=False)

        # Initialize nearfield phase.
        self.phase = None
        self.reset_phase(phase)

        # Initialize everything else inside reset.
        self.reset(reset_phase=False, reset_flags=False)

        # Custom GPU kernels for speedy weighting.
        self._update_weights_generic_cuda_kernel = None
        if np != cp and False:  # Disabled until 0.1.1
            try:
                self._update_weights_generic_cuda_kernel = cp.RawKernel(
                    CUDA_KERNELS, "update_weights_generic"
                )
            except:
                pass

    # Initialization helper functions.
    def reset(self, reset_phase=True, reset_flags=False):
        r"""
        Resets the hologram to an initial state. Does not restore the preconditioned ``phase``
        that may have been passed to the constructor (as this information is lost upon
        optimization). Instead, phase is randomized if ``reset_phase=True``.
        Also uses the current ``target`` rather than the ``target`` that may have been
        passed to the constructor (e.g. includes current
        :meth:`~slmsuite.holography.algorithms.SpotHologram.refine_offset()` changes, etc).

        Parameters
        ----------
        reset_phase : bool
            Whether to additionally call :meth:`reset_phase()`.
        reset_flags : bool:
            Whether to erase the information (including passed ``kwargs``) stored in :attr:`flags`.
        """
        # Reset phase to random if needed.
        if self.phase is None or reset_phase:
            self.reset_phase()

        # Reset weights.
        self.reset_weights()

        # Reset vars.
        self.iter = 0
        self.stats = {"method": [], "flags": {}, "stats": {}}
        if reset_flags:
            self.flags = {"method": ""}

        # Reset farfield storage.
        self.amp_ff = None
        self.phase_ff = None

        # Reset complex looping variables.
        self.nearfield = cp.zeros(self.shape, dtype=self.dtype_complex)
        if not self.target is None:
            self.farfield = cp.zeros(self.target.shape, dtype=self.dtype_complex)

    def _get_target_moments_knm_norm(self):
        """
        Get the first and second order moments of the target in normalized knm space
        (knm integers divided by shape)
        """
        # Grab the target.
        target = self.target
        if hasattr(target, "get"):
            target = self.target.get()

        # Figure out the size of the target in knm space
        center_knm = analysis.image_positions(
            target, nansum=True
        )  # Note this is centered knm space.

        # FUTURE: handle shear.
        std_knm = np.sqrt(
            analysis.image_variances(target, centers=center_knm, nansum=True)[:2, 0]
        )

        # Normalized knm divides by the shape. This is such that these values can be
        # compared even for holograms of different shape.
        shape = np.flip(self.shape).astype(float)

        return np.squeeze(center_knm) / shape, np.squeeze(std_knm) / shape

    def _get_quadratic_initial_phase(self, scaling=1):
        """
        Analytically guesses a phase pattern (lens, blaze) that will overlap with the target.
        """
        if hasattr(self.amp, "get"):
            std_amp = np.sqrt(analysis.image_variances(self.amp.get())[:2, 0])
        else:
            std_amp = np.sqrt(analysis.image_variances(self.amp)[:2, 0])
        slm_shape = np.flip(self.slm_shape).astype(float)
        std_amp /= slm_shape

        center_knm_norm, std_knm_norm = self._get_target_moments_knm_norm()

        grid = analysis._generate_grid(
            self.slm_shape[1], self.slm_shape[0], centered=True
        )
        grid = [grid[0].astype(self.dtype), grid[1].astype(self.dtype)]
        grid[0] /= self.slm_shape[1]
        grid[1] /= self.slm_shape[0]

        # Figure out what lens and blaze we should apply to initialize to cover
        # the target, based upon the moments we calculated.
        return cp.array(
            tphase.blaze(grid, slm_shape * center_knm_norm)
            + tphase.lens(
                grid, np.reciprocal(scaling * slm_shape * std_knm_norm / std_amp)
            ),
            dtype=self.dtype,
            copy=(False if np.__version__[0] == "1" else None),
        )

    def _get_random_phase(self):
        if cp == np:  # numpy does not support `dtype=`
            rng = np.random.default_rng()
            return rng.uniform(-np.pi, np.pi, self.slm_shape).astype(self.dtype)
        else:
            return cp.random.uniform(-np.pi, np.pi, self.slm_shape, dtype=self.dtype)

    def reset_phase(self, custom_phase=None, random_phase=None, quadratic_phase=None):
        r"""
        Resets the hologram
        to a provided phase,
        to a random state,
        or to a `quadratic phase <https://doi.org/10.1364/OE.16.002176>`_
        which overlaps with the target pattern.

        Parameters
        ----------
        custom_phase : array_like OR None
            Custom nearfield initial phase. If not ``None``, then all other parameters
            are ignored.
            See :attr:`phase`. :attr:`phase` should only be passed if the user wants to
            precondition the optimization. Of shape :attr:`slm_shape`.
        random_phase : float OR None
            Sets the phase to uniformly random phase, scaled to :math:`2\pi`.
            Setting ``random_phase`` to a fraction of 1 likewise scales the randomness.
            If ``None``, looks for ``"random_phase"`` in :attr:`flags`.
            This adds with the ``quadratic_phase`` parameter.
        quadratic_phase : bool OR float OR None
            We can also precondition the phase analytically (with a lens and blaze)
            to roughly the size of the target hologram, according to the first and
            second order :meth:`~slmsuite.holography.analysis.image_moments()`.
            This quadratic preconditioning is
            `thought to help reduce the formation of optical vortices or speckle
            <https://doi.org/10.1364/OE.16.002176>`_
            compared to random initialization, as the analytic distribution
            is smooth in phase.
            If ``None``, looks for ``"quadratic_phase"`` in :attr:`flags`.
            If a ``float`` is provided, the size of the beam in the
            farfield is scaled accordingly.
            This feature is ignored if ``phase`` is not ``None``.
        """
        if self.phase is None:
            self.phase = cp.zeros(self.slm_shape, dtype=self.dtype)

        if custom_phase is not None:
            custom_phase = cp.array(
                custom_phase,
                dtype=self.dtype,
                copy=(False if np.__version__[0] == "1" else None),
            )

            if not np.all(np.array(self.slm_shape) == np.array(custom_phase.shape)):
                raise ValueError(
                    f"Reset phase of shape {custom_phase.shape} is not of slm_shape {self.slm_shape}"
                )

            cp.copyto(self.phase, custom_phase)
        else:
            # Parse quadratic_phase
            if quadratic_phase is None:
                if "quadratic_phase" in self.flags:
                    quadratic_phase = self.flags["quadratic_phase"]
                else:
                    quadratic_phase = False

            # Parse quadratic_phase
            if random_phase is None:
                if "random_phase" in self.flags:
                    random_phase = self.flags["random_phase"]
                else:
                    random_phase = 1

            self.phase.fill(0)

            # Reset phase to random if no custom_phase is given.
            if quadratic_phase:  # Analytic
                self.phase += self._get_quadratic_initial_phase(quadratic_phase)
            if random_phase:  # Random
                self.phase += random_phase * self._get_random_phase()

    def reset_weights(self):
        """
        Resets the hologram weights to the :attr:`target` defaults.
        """
        # Copy from the target.
        self.weights = self.target.copy()

        if hasattr(self, "zero_weights"):
            self.zero_weights *= 0

        # Account for MRAF by setting any noise region to zero by default.
        cp.nan_to_num(self.weights, copy=False, nan=0)

    @staticmethod
    def get_padded_shape(
        slm_shape,
        padding_order=1,
        square_padding=True,
        precision=np.inf,
        precision_basis="kxy",
    ):
        """
        Helper function to calculate the shape of the computational space.
        For a given base ``slm_shape``, pads to the user's requirements.
        If the user chooses multiple requirements, the largest
        dimensions for the shape are selected.
        By default, pads to the smallest square power of two that
        encapsulates the original ``slm_shape``.

        See Also
        ~~~~~~~~
        :class:`Hologram` for more information about the importance of padding.

        Note
        ~~~~
        Under development: a parameter to pad based on available memory
        (see :meth:`_calculate_memory_constrained_shape()`).

        Parameters
        ----------
        slm_shape : (int, int) OR slmsuite.hardware.FourierSLM
            The original shape of the SLM in :mod:`numpy` `(h, w)` form. The user can pass a
            :class:`~slmsuite.hardware.FourierSLM` or :class:`~slmsuite.hardware.SLM` instead,
            and should pass this when using the ``precision`` parameter.
        padding_order : int
            Scales to the ``padding_order`` th larger power of 2.
            A ``padding_order`` of zero does nothing. For instance, an SLM
            with shape ``(720, 1280)`` would yield
            ``(720, 1280)`` for ``padding_order=0``,
            ``(1024, 2048)`` for ``padding_order=1``, and
            ``(2048, 4096)`` for ``padding_order=2``.
        square_padding : bool
            If ``True``, sets the smaller shape dimension to that of the larger, yielding a square.
        precision : float
            Returns the shape that produces a computational k-space with resolution smaller
            than ``precision``. The default, infinity, requests a padded shape larger
            than zero, so ``padding_order`` will dominate.
        precision_basis : str
            Basis for the precision. Can be ``"ij"`` (camera) or ``"kxy"`` (normalized blaze).

        Returns
        -------
        (int, int)
            Shape of the computational space which satisfies the above requirements.
        """
        cameraslm = None
        # If slm_shape is actually a FourierSLM.
        if hasattr(slm_shape, "slm") and hasattr(slm_shape, "cam"):
            cameraslm = slm_shape
            slm_shape = cameraslm.slm.shape

        # If slm_shape is actually a SLM.
        elif hasattr(slm_shape, "shape"):
            cameraslm = lambda: 0  # Make a fake cameraslm
            cameraslm.slm = slm_shape  # At this point, slm_shape is the SLM.
            slm_shape = (
                cameraslm.slm.shape
            )  # And make the shape variable actually the shape.

            if precision_basis == "ij":
                raise ValueError(
                    "Must pass a CameraSLM object under slm_shape "
                    "to use the 'ij' precision_basis!"
                )

        # Handle precision.
        if np.isfinite(precision) and cameraslm is not None:
            if precision <= 0:
                raise ValueError(
                    "Precision passed to get_padded_shape() must be positive."
                )
            dpixel = np.amin(cameraslm.slm.pitch)
            fs = 1 / dpixel  # Sampling frequency

            if precision_basis == "ij":
                slm_range = np.amax(cameraslm.kxyslm_to_ijcam([fs, fs]))
                pixels = slm_range / precision
            elif precision_basis == "kxy":
                pixels = fs / precision

            # Raise to the nearest greater power of 2.
            pixels = np.power(2, int(np.ceil(np.log2(pixels))))
            precision_shape = (pixels, pixels)
        elif np.isfinite(precision):
            raise ValueError(
                "Must pass a CameraSLM object under slm_shape "
                "to implement get_padded_shape() precision calculations!"
            )
        else:
            precision_shape = slm_shape

        # Handle padding_order.
        if padding_order > 0:
            padding_shape = np.power(
                2, np.ceil(np.log2(slm_shape)) + padding_order - 1
            ).astype(int)
        else:
            padding_shape = slm_shape

        # Take the largest and square if desired.
        shape = tuple(np.amax(np.vstack((precision_shape, padding_shape)), axis=0))

        if square_padding:
            largest = np.amax(shape)
            shape = (largest, largest)

        return shape

    def _calculate_memory_constrained_shape(self, device=0, dtype=None):
        if dtype is None:
            dtype = self.dtype

        memory = Hologram.get_mempool_limit(device=device)

        num_values = memory / dtype(0).nbytes

        # (4 real arrays, 2 complex arrays for DFT holograms)
        num_values_per_array = num_values / 8

        return np.sqrt(num_values_per_array)

    # User interactions: Changing the target and recovering the nearfield phase and complex farfield.
    def _set_target(self, new_target, reset_weights=False):
        """
        Change the target to something new. This method handles cleaning and normalization.

        This method is shelled by :meth:`set_target()` such that it is still accessible
        in the case that a subclass overwrites :meth:`set_target()`.

        Tip
        ~~~
        Use :meth:`.plot_farfield()` on :attr:`target`` for visualization.

        Parameters
        ----------
        new_target : array_like OR None
            If ``None``, sets the target to zero. The ``None`` case is used internally
            by :class:`SpotHologram`.
        reset_weights : bool
            Whether to overwrite ``weights`` with ``target``.
        """
        if new_target is None:
            self.target = cp.zeros(shape=self.target_shape, dtype=self.dtype)
        else:
            self.target = cp.array(
                new_target,
                dtype=self.dtype,
                copy=(False if np.__version__[0] == "1" else None),
            )
            cp.abs(self.target, out=self.target)
            with warnings.catch_warnings():
                self.target *= 1 / Hologram._norm(self.target)

        if reset_weights:
            self.reset_weights()

    def set_target(self, new_target, reset_weights=False):
        """
        Change the target to something new. This method handles cleaning and normalization.

        Parameters
        ----------
        new_target : array_like OR None
            New :attr:`target` to optimize towards. Should be of shape :attr:`shape`.
            If ``None``, :attr:`target` is zeroed (used internally, but probably should not
            be used by a user).
        reset_weights : bool
            Whether to update the :attr:`weights` to this new :attr:`target`.
        """
        self._set_target(new_target=new_target, reset_weights=reset_weights)

    def get_phase(self, include_propagation=False):
        r"""
        Collects the current nearfield phase from the GPU with :meth:`cupy.ndarray.get()`.
        Also shifts the :math:`[-\pi, \pi]` range of :meth:`numpy.arctan2()` to :math:`[0, 2\pi]`
        for faster writing to the SLM (see :meth:`~slmsuite.hardware.slms.slm.SLM.set_phase()`).

        Parameters
        ----------
        include_propagation : bool
            Whether to include the :attr:`propagation_kernel`, if available.

        Returns
        -------
        numpy.ndarray
            Current nearfield phase of the optimization.
        """
        if include_propagation and self.propagation_kernel is not None:
            if cp != np:
                return (self.phase + self.propagation_kernel).get()
            else:
                return self.phase + self.propagation_kernel
        else:
            if cp != np:
                return self.phase.get() + np.pi
            else:
                return self.phase + np.pi

    def get_farfield(self, shape=None, propagation_kernel=None, affine=None, get=True):
        r"""
        Collects the current complex DFT farfield, potentially with transformations.
        This includes collecting the data from the GPU with :meth:`cupy.ndarray.get()`.

        Parameters
        ----------
        shape : (int, int)
            Shape of the DFT.
            Useful to change the resolution of the farfield.
            If ``None``, defaults to :attr:`shape`, and falls back to :attr:`slm_shape`.
        propagation_kernel : array_like
            Used to check the result of the hologram at different depths.
            See :attr:`propagation_kernel`. If ``None``, defaults to
            :attr:`propagation_kernel` if one is present. Otherwise, no kernel is
            applied. Zeroing can force no kernel to be applied and yield the raw DFT.
        affine : dict
            Affine transformation to apply to farfield data (in the form of a dictionary
            with keys ``"M"`` and ``"b"``).
            If ``None``, no transformation is applied.
        get : bool
            Whether or not to convert the cupy array to a numpy array if cupy is used.
            This is ignored if numpy is used.

        Returns
        -------
        numpy.ndarray
            Current farfield expected from the current :attr:`phase`.
        """
        # Parse shape.
        if shape is None:
            shape = self.shape
        if len(shape) == 1:
            shape = self.slm_shape

        # Parse propagation_kernel
        if propagation_kernel is None:
            propagation_kernel = self.propagation_kernel
        if propagation_kernel is None:
            propagation_kernel = 0
        if not np.isscalar(propagation_kernel):
            propagation_kernel = cp.array(
                propagation_kernel, copy=(False if np.__version__[0] == "1" else None)
            )

        # This doesn't use self.nearfield, self.farfield because we might be using different shape.
        nearfield = toolbox.pad(
            self.amp * cp.exp(1j * (self.phase + propagation_kernel)), shape
        )
        farfield = cp.fft.fftshift(
            cp.fft.fft2(cp.fft.fftshift(nearfield), norm="ortho")
        )

        # Only populate amp and phase if
        if self.amp_ff is not None and shape == self.amp_ff.shape:
            self.amp_ff = cp.abs(farfield, out=self.amp_ff)
            self.phase_ff = cp.arctan2(farfield.imag, farfield.real, out=self.phase_ff)

        # Transform as desired. Note that this will likely break normalization.
        if cp != np:
            if affine is not None:
                cp_affine_transform(
                    input=farfield,
                    matrix=affine["M"],
                    offset=affine["b"],
                    output_shape=shape,
                    order=3,
                    output=farfield,
                    mode="constant",
                    cval=0,
                )
            return farfield.get() if get else farfield
        else:
            if affine is not None:
                sp_affine_transform(
                    input=farfield,
                    matrix=affine["M"],
                    offset=affine["b"],
                    output_shape=shape,
                    order=3,
                    output=farfield,
                    mode="constant",
                    cval=0,
                )
            return farfield

    # Propogation: nearfield-farfield helper functions.
    def _populate_results(self):
        """
        Helper function to populate:
            - farfield
            - amp_ff
            - phase_ff

        From the data in:
            - amp
            - phase

        Pass a nearfield complex matrix of self.shape shape to avoid memory reallocation.
        """
        self._nearfield2farfield()
        self.amp_ff = cp.abs(self.farfield, out=self.amp_ff)
        self.phase_ff = cp.arctan2(
            self.farfield.imag, self.farfield.real, out=self.phase_ff
        )

    def _midloop_cleaning(self):
        # 2.1) Cache amp_ff for weighting (if None, will init; otherwise in-place).
        self.amp_ff = cp.abs(self.farfield, out=self.amp_ff)

        # 2.2) Erase images from the past loop. FUTURE: Make better and faster.
        if hasattr(self, "img_ij"):
            self.img_ij = None
        if hasattr(self, "img_knm"):
            self.img_knm = None

    def remove_vortices(self, plot=False):
        """
        Removes the computed phase vortices in the farfield where the target amplitude is positive.
        Useful for smoothing out the pattern and reducing speckle.
        The user can call this method by passing a ``callback=`` function containing it.
        For instance:

        .. code-block:: python

            # Define a function to use a callback.
            def remove_vortices_callback(holo):
                if holo.iter % 10 == 9:     # Only remove vortices every 10 iterations.
                    holo.remove_vortices()  # This method is slightly expensive, so calling every loop is not advised.

            # The function will be called during the loop.
            hologram.optimize(..., callback=remove_vortices_callback)

        Important
        ~~~~~~~~~
        This callback can only applied be during a GS loop. To use for a conjugate
        gradient hologram, do a single iteration of GS.

        Parameters
        ----------
        plot : bool
            Enable debug plots.
        """
        if self.phase_ff is not None:
            limits = self.plot_farfield(self.target)

            if plot:
                self.plot_farfield(self.phase_ff, title="phase original", limits=limits)
                self.plot_farfield(
                    analysis.image_vortices(self.phase_ff),
                    title="vortices coords",
                    limits=limits,
                )
                self.plot_farfield(
                    (self.target > 0).astype(float), title="target_mask", limits=limits
                )
                self.plot_farfield(
                    (self.target > 0).astype(float)
                    + analysis.image_vortices(self.phase_ff),
                    title="vortices coords + target_mask",
                    limits=limits,
                )
                self.plot_farfield(
                    analysis.image_vortices_remove(
                        self.phase_ff, self.target > 0, True
                    ),
                    title="phase vortices",
                    limits=limits,
                )
                analysis.image_vortices_remove(self.phase_ff, self.target > 0)
                self.plot_farfield(
                    self.phase_ff, title="phase removal after", limits=limits
                )

    def _build_nearfield(self, phase_torch=None):
        """Populate nearfield with data from amp and phase."""
        (i0, i1, i2, i3) = toolbox.unpad(self.shape, self.slm_shape)
        self.nearfield.fill(0)

        if phase_torch is None:
            if self.propagation_kernel is None:
                self.nearfield[i0:i1, i2:i3] = self.amp * cp.exp(1j * self.phase)
            else:
                self.nearfield[i0:i1, i2:i3] = self.amp * cp.exp(
                    1j * (self.phase + self.propagation_kernel)
                )

            return self.nearfield
        else:
            nearfield_torch = self._get_torch_tensor_from_cupy(self.nearfield)
            amp_torch = self._get_torch_tensor_from_cupy(self.amp)
            prop_torch = self._get_torch_tensor_from_cupy(self.propagation_kernel)

            self.optimizer.zero_grad()

            if prop_torch is None:
                nearfield_torch[i0:i1, i2:i3] = amp_torch * torch.exp(1j * phase_torch)
            else:
                nearfield_torch[i0:i1, i2:i3] = amp_torch * torch.exp(
                    1j * (phase_torch + prop_torch)
                )

            return nearfield_torch

    def _nearfield_extract(self):
        """Populate phase with data from nearfield."""
        (i0, i1, i2, i3) = toolbox.unpad(self.shape, self.slm_shape)

        self.phase = cp.arctan2(
            self.nearfield.imag[i0:i1, i2:i3],
            self.nearfield.real[i0:i1, i2:i3],
            out=self.phase,
        )
        if self.propagation_kernel is not None:
            self.phase -= self.propagation_kernel

    def _nearfield2farfield(self, phase_torch=None):
        """
        Maps the nearfield to the farfield by a discrete Fourier transform.
        This should populate :attr:`farfield`.
        This function is overloaded by subclasses.
        """
        # This may return a torch nearfield if we are in torch mode.
        nearfield = self._build_nearfield(phase_torch)

        if phase_torch is None:
            self.farfield = cp.fft.fftshift(
                cp.fft.fft2(cp.fft.fftshift(nearfield), norm="ortho")
            )
        else:
            farfield_torch = self._get_torch_tensor_from_cupy(self.farfield)
            farfield_torch = torch.fft.fftshift(
                torch.fft.fft2(torch.fft.fftshift(nearfield), norm="ortho")
            )
            self.farfield = cp.asarray(farfield_torch.detach())

            return farfield_torch

        self._midloop_cleaning()

    def _farfield2nearfield(self, extract=True):
        """
        Maps the farfield to the nearfield by a discrete Fourier transform.
        This should populate populate :attr:`nearfield`.
        This function is overloaded by subclasses.

        Parameters
        ----------
        extract : bool
            Whether to extract data into the :attr:`phase` variable. This is not used
            for :class:`MultiplaneHologram`.
        """
        self.nearfield = cp.fft.ifftshift(
            cp.fft.ifft2(cp.fft.ifftshift(self.farfield), norm="ortho")
        )

        if extract:
            self._nearfield_extract()

    # Core optimization function.
    def optimize(
        self,
        method="GS",
        maxiter=20,
        verbose=True,
        callback=None,
        feedback=None,
        stat_groups=[],
        **kwargs,
    ):
        r"""
        Optimizers to solve the "phase problem": approximating the nearfield phase that
        transforms a known nearfield source amplitude to a desired farfield
        target amplitude.
        Supported optimization methods include:

        -   Gerchberg-Saxton (GS) phase retrieval.

            - ``'GS'``

              `An iterative algorithm for phase retrieval
              <http://www.u.arizona.edu/~ppoon/GerchbergandSaxton1972.pdf>`_,
              accomplished by moving back and forth between the imaging and Fourier domains,
              with amplitude corrections applied to each.
              This is usually implemented using fast discrete Fourier transforms,
              potentially GPU-accelerated.

        -   Weighted Gerchberg-Saxton (WGS) phase retrieval algorithms of various flavors.
            Improves the uniformity of GS-computed focus arrays using weighting methods and
            techniques from literature. The ``method`` keywords are:

            - ``'WGS-Leonardo'``

              `The original WGS algorithm <https://doi.org/10.1364/OE.15.001913>`_.
              Weights the target amplitudes by the ratio of mean amplitude to computed
              amplitude, which amplifies weak spots while attenuating strong spots. Uses
              the following weighting function:

              .. math:: \mathcal{W} = \mathcal{W}\left(\frac{\mathcal{T}}{\mathcal{F}}\right)^p

              where :math:`\mathcal{W}`, :math:`\mathcal{T}`, and :math:`\mathcal{F}`
              are the weight amplitudes,
              target (goal) amplitudes, and
              feedback (measured) amplitudes,
              and :math:`p` is the power passed as ``"feedback_exponent"`` in
              :attr:`~slmsuite.holography.algorithms.Hologram.flags` (see ``kwargs``).
              The power :math:`p` defaults to .8 if not passed. In general, smaller
              :math:`p` will lead to slower yet more stable optimization.

            - ``'WGS-Kim'``

              `Improves the convergence <https://doi.org/10.1364/OL.44.003178>`_
              of ``WGS-Leonardo`` by fixing the farfield phase
              strictly after a desired number of net iterations
              specified by ``"fix_phase_iteration"``
              or after exceeding a desired efficiency
              (fraction of farfield energy at the desired points)
              specified by ``"fix_phase_efficiency"``

            - ``'WGS-Nogrette'``

              Weights target intensities by `a tunable gain factor <https://doi.org/10.1103/PhysRevX.4.021034>`_.

              .. math:: \mathcal{W} = \mathcal{W}/\left(1 - f\left(1 - \mathcal{F}/\mathcal{T}\right)\right)

              where :math:`f` is the gain factor passed as ``"feedback_factor"`` in
              :attr:`~slmsuite.holography.algorithms.Hologram.flags` (see ``kwargs``).
              The factor :math:`f` defaults to .1 if not passed.

              Note that while Nogrette et al compares powers, this implementation
              compares amplitudes for speed. These are identical to first order.

            - ``'WGS-Wu'``

              Weights using an `exponential <https://doi.org/10.1364/OE.413723>`_
              function, which is less sensitive to near-zero values of
              :math:`\mathcal{F}` or :math:`\mathcal{T}`.

              .. math:: \mathcal{W} = \mathcal{W}\exp\left( p (\mathcal{T} - \mathcal{F}) \right)

              The speed of correction is controlled by :math:`p`,
              the power passed as ``"feedback_exponent"``.

            - ``'WGS-tanh'``

              Weights by hyperbolic tangent, commonly used as an
              `activation function
              <https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions>`_
              in machine learning.

              .. math:: \mathcal{W} = \mathcal{W}\left[1 + f\text{tanh}\left( p (\mathcal{T} - \mathcal{F}) \right) \right]

              This weighting limits each update to a relative change of :math:`\pm f`,
              passed as ``"feedback_factor"``, which is useful to prevent large changes.
              The speed of correction is controlled by :math:`p`,
              the power passed as ``"feedback_exponent"``.

        -   Conjugate Gradient (CG) phase retrieval.

            - ``'CG'``

              **(This feature is experimental.)**

              Some holography---especially that with more complicated holographic
              objectives---can be better treated with gradient-based methods.
              In these cases, the phase is guided to an optimized state by following the
              `back-propogated <https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html>`_
              gradients (with respect to phase) of given objective ``loss`` which is
              passed as one of the :attr:`flags` to :meth:`.optimize()`.
              Weighting different components of the objective leads to tradeoffs between
              those components: for instance a tradeoff between power guided into a given
              pattern and the uniformity of the realized pattern.
              :mod:`slmsuite` uses :mod:`pytorch` as a backend for gradient computation.
              Notably, memory is still owned and initialized by :mod:`cupy`, but
              gradients can be calculated by using :mod:`pytorch`-:mod:`cupy`
              `interoperability <https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch>`_.

              The objective ``loss`` is expected to be a :class:`torch.nn.Module`
              and defaults to a complex variant of ``torch.nn.MSELoss()``.
              ``loss`` is called in the style of :mod:`pytorch`, using (as arguments)
              the computed ``farfield`` (with gradient tree intact) and
              the ``target`` values for the farfield. Internally, this looks like:

              .. code-block:: python

                result = loss(      # The user provides this nn.Module to .optimize()
                    farfield,       # The farfield (with gradients), calculated from `phase` by slmsuite
                    target          # The target, initialized by the user and processed by slmsuite
                )
                result.backward()   # Gradients are back-propagated to the input `phase`.

              For :class:`~slmsuite.holography.algorithms.FeedbackHologram` and
              subclasses, the gradients are computed computationally, but the
              computational values are then replaced with the experimental results.
              This allows optimization of the experimental results using the
              computational gradients (correct to first order) as a guide.
              Currently, feedback is *not supported* for spot arrays with
              ``"experimental_spot"`` or ``"computational_spot"`` feedback
              (WGS probably works better for such spot array objectives anyway).

              Creating a custom objective is as simple as making a custom
              :meth:`torch.nn.Module.forward()` method.
              These methods can be as simple as
              `a single expression <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`_
              or as complicated as
              `a full neural network <https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html>`_
              operating on the input parameters.
              However, remember to use :mod:`pytorch` methods because the arguments are
              of type :class:`torch.Tensor`.
              Here's an example of a custom :meth:`torch.nn.Module.forward()`
              which implements the `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_:

              .. code-block:: python

                # Define the loss as a class.
                class HuberLoss(nn.Module):
                    def __init__(self, delta=1.0):
                        super(HuberLoss, self).__init__()
                        self.delta = delta

                    def forward(self, farfield, target):
                        residual = torch.abs(farfield - target)
                        quadratic = torch.clamp(residual, max=self.delta)
                        linear = residual - quadratic
                        loss = 0.5 * quadratic ** 2 + self.delta * linear

                        return torch.mean(loss)

                # Initialize the class. Remember that we can pass arguments (delta) here.
                loss = HuberLoss(delta=2.0)

                # Pass the loss to the hologram by one of two methods:
                hologram.optimize(..., loss=loss)       # 1. Pass as **kwarg.
                hologram.flags["loss"] = loss           # 2. Set directly.

              MRAF (next section), if desired, needs to be handled by the ``loss`` function.
              MRAF information is encoded in the ``target``, with the noise region being ``nan``.

        -   The option for `Mixed Region Amplitude Freedom (MRAF)
            <https://doi.org/10.1007/s10043-018-0456-x>`_ feedback. In standard
            iterative algorithms, the entire Fourier-domain unpatterned field is replaced with zeros.
            This is disadvantageous because a desired farfield pattern might not be especially
            compatible with a given nearfield amplitude, or otherwise. MRAF enables
            "noise regions" where some fraction of the given farfield is **not** replaced
            with zeros and instead is allowed to vary. In practice, MRAF is
            enabled by setting parts of the :attr:`target` to ``nan``; these regions act
            as the noise regions. The ``"mraf_factor"`` flag in
            :attr:`~slmsuite.holography.algorithms.Hologram.flags`
            allows the user to partially attenuate the noise regions.
            A factor of 0 fully attenuates the noise region (normal WGS behavior).
            A factor of 1 does not attenuate the noise region at all (the default).
            Middle ground is recommended, but is application-dependent as a
            tradeoff between improving pattern fidelity and maintaining pattern efficiency.

            As examples, consider two cases where MRAF can be useful:

            - **Sloping a top hat.** Suppose we want very flat amplitude on a beam.
              Requesting a sharp edge to this beam can lead to fringing effects at the
              boundary which mitigate flatness both inside and outside the beam.
              If instead a noise region is defined in a band surrounding the beam,
              the noise region will be filled with whatever slope best enables the
              desired flat beam.

            - **Mitigating diffractive orders.** Without MRAF, spot patterns with high
              crystallinity often have "ghost" diffractive orders which continue the
              pattern past the edges of requested spots. Even though these orders are attenuated
              during each phase retrieval iteration, they remain part of the best
              solution for the recovered phase. With MRAF, a noise region can help solve
              for retrieved phase which does not generate these undesired orders.

        Caution
        ~~~~~~~
        Requesting ``stat_groups`` will slow the speed of optimization due to the
        overhead of processing and saving statistics, especially in the case of
        GPU-accelerated optimization where significant time cost is incurred by
        moving these statistics to the CPU. This is especially apparent in the case
        of fully-computational holography, where this effect can slow what is otherwise
        a fully-GPU-contained loop by an order magnitude.

        Tip
        ~~~
        This function uses a parameter naming convention borrowed from
        :meth:`scipy.optimize.minimize()` and other functions in
        :mod:`scipy.optimize`. The parameters ``method``, ``maxiter``, and ``callback``
        have the same functionality as the equivalently-named parameters in
        :meth:`scipy.optimize.minimize()`.

        Parameters
        ----------
        method : str
            Optimization method to use. See the list of optimization methods above.
        maxiter : int
            Number of iterations to optimize before terminating.
        verbose : bool OR int
            Whether to display :mod:`tqdm` progress bars.
            These bars are also not displayed for ``maxiter <= 1``.
            If ``verbose`` is greater than 1, then flags are printed as a preamble.
        callback : callable OR None
            Same functionality as the equivalently-named parameter in
            :meth:`scipy.optimize.minimize()`. ``callback`` must accept a Hologram
            or Hologram subclass as the single argument. If ``callback`` returns
            ``True``, then the optimization exits. Ignored if ``None``.
        feedback : str OR None
            Type of feedback to use during optimization, for instance when weighting in ``"WGS"``.
            For direct instances of :class:`Hologram`, this can only
            be ``"computational"`` feedback. Subclasses support more types of feedback.
            Supported feedback options include the following:

            - ``"computational"`` Uses the the projected farfield pattern (transform of
              the complex nearfield) as feedback.
            - ``"experimental"`` Uses a camera contained in a passed ``cameraslm`` as feedback.
              Specific to subclasses of :class:`FeedbackHologram`.
            - ``"computational_spot"`` Takes the computational result (the projected
              farfield pattern) and integrates regions around the expected positions of
              spots in an optical focus array. More stable than ``"computational"`` for spots.
              Specific to subclasses of :class:`SpotHologram`.
            - ``"experimental_spot"`` Takes the experimental result (the image from a
              camera) and integrates regions around the expected positions of
              spots in an optical focus array. More stable than ``"experimental"`` for spots.
              Specific to subclasses of :class:`SpotHologram`.
            - ``"external_spot"`` Uses some external user-provided metric for spot
              feedback. See :attr:`external_spot_amp`.
              Specific to subclasses of :class:`SpotHologram`.

        stat_groups : list of str OR None
            Strings representing types of feedback (data gathering) upon which
            statistics should be derived. These strings correspond to valid types of
            feedback (see above). For instance, if ``"experimental"`` is passed as a
            stat group, statistics on the pixels in the experimental feedback image will
            automatically be computed and stored for each iteration of optimization.
            However, this comes with overhead (see above warning).
        **kwargs : dict, optional
            Various weight keywords and values to pass depending on the weight method.
            These are passed into :attr:`flags`. See options documented in the constructor.
        """
        # 1) Update flags based upon the arguments.
        self._update_flags(method, verbose, feedback, stat_groups, **kwargs)

        # 2) Prepare the iterations iterable.
        iterations = range(maxiter)

        # 2.1) Decide whether to use a tqdm progress bar. Don't use a bar for maxiter == 1.
        if verbose and maxiter > 1:
            iterations = tqdm(iterations)

        # 3) Switch between optimization methods (currently only GS- or WGS-type is supported).
        if "GS" in method:
            self.optimize_gs(iterations, callback)
        elif "CG" in method:
            self.optimize_cg(iterations, callback)
        else:
            raise ValueError(f"Unsupported optimization method '{method}'")

    def _update_flags(self, method, verbose, feedback, stat_groups, **kwargs):
        """
        Helper function for :meth:`optimize()` to parse arguments.
        """
        # 0) Check and record method.
        methods = list(ALGORITHM_DEFAULTS.keys())
        if not method in methods:
            raise ValueError(
                "Unrecognized method '{}'.\n"
                "Valid methods include {}".format(method, methods)
            )
        self.flags["method"] = method

        # 1) Parse flags:
        # 1.1) Set defaults if not already set.
        for flag, value in ALGORITHM_DEFAULTS[method].items():
            if not flag in self.flags:
                self.flags[flag] = value
        if not "fixed_phase" in self.flags:
            self.flags["fixed_phase"] = False

        # 1.2) Parse kwargs as flags.
        for flag in kwargs:
            self.flags[flag] = kwargs[flag]

        # 1.3) Add in non-defaulted flags, with error checks
        for group in stat_groups:
            if not (group in FEEDBACK_OPTIONS):
                raise ValueError(
                    "Statistics group '{}' not recognized as a feedback option.\n"
                    "Valid options: {}".format(group, FEEDBACK_OPTIONS)
                )
        self.flags["stat_groups"] = stat_groups

        if feedback is not None:
            if not (feedback in FEEDBACK_OPTIONS):
                raise ValueError(
                    "Feedback '{}' not recognized as a feedback option.\n"
                    "Valid options: {}".format(group, FEEDBACK_OPTIONS)
                )
            self.flags["feedback"] = feedback

        # 1.4) Print the flags if verbose.
        if verbose > 1:
            print(
                f"Optimizing with '{method}' using the following method-specific flags:"
            )
            pprint.pprint(
                {
                    key: value
                    for (key, value) in self.flags.items()
                    if key in ALGORITHM_DEFAULTS[method]
                }
            )
            print("", end="", flush=True)  # Prevent tqdm conflicts.

    # GS- or WGS-type optimization.
    def optimize_gs(self, iterations, callback):
        """
        GPU-accelerated Gerchberg-Saxton (GS) iterative phase retrieval.

        Solves the "phase problem": approximates the nearfield phase that
        transforms a known nearfield source amplitude to a desired farfield
        target amplitude.

        Caution
        ~~~~~~~
        This function should be called through :meth:`.optimize()` and not called
        directly. It is left as a public function exposed in documentation to clarify
        how the internals of :meth:`.optimize()` work.

        Note
        ~~~~
        Default FFTs are **not** in-place in this algorithm. In both non-:mod:`cupy` and
        :mod:`cupy` implementations, :mod:`numpy.fft` does not support in-place
        operations.  However, :mod:`scipy.fft` does in both. In the future, we may move to the scipy
        implementation. However, neither :mod:`numpy` or :mod:`scipy` ``fftshift`` support
        in-place movement (for obvious reasons). For even faster computation, algorithms should
        consider **not shifting** the FFT result, and instead shifting measurement data / etc to
        this unshifted basis. We might also implement `get_fft_plan
        <https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.fftpack.get_fft_plan.html>`_
        for even faster FFTing. However, in practice, speed is limited by other
        peripherals (especially feedback and stats) rather than FFT speed or memory.

        Parameters
        ----------
        iterations : iterable
            Number of loop iterations to run. Is an iterable to pass a :mod:`tqdm` iterable.
        callback : callable OR None
            See :meth:`.optimize()`.
        """
        # Precompute MRAF helper variables.
        # In particular, this stores the binary masks for the signal, noise, and null regions.
        mraf_variables = self._mraf_helper_routines()

        for _ in iterations:
            # (A) Nearfield -> Farfield
            # This uses the self.phase and self.amplitude attributes to populate the
            # self.farfield attribute. Also cleans per-loop variables such as self.img_ij
            self._nearfield2farfield()

            # (B) Midloop Farfield Routines
            # (B.1) Run step function if present and check termination conditions.
            if callback is not None:
                if callback(self):
                    break

            # (B.2) Update statistics based on the current farfield and potentially current
            # experimental results.
            self._update_stats(self.flags["stat_groups"])

            # (B.3) Evaluate method-specific routines, stats, etc. This includes camera feedback/etc.
            # If you want to add new functionality to GS, do so here to keep the main loop clean.
            self._gs_farfield_routines(mraf_variables)

            # (C) Farfield -> Nearfield
            # This populates the self.nearfield and self.phase attributes.
            self._farfield2nearfield()

            # Increment iteration.
            self.iter += 1

        # Update the final farfield using phase and amp.
        self._populate_results()

    def _mraf_helper_routines(self):
        # MRAF helper variables
        if np == cp:
            mraf_enabled = np.isnan(np.sum(self.target))
        else:
            mraf_enabled = np.isnan(cp.sum(self.target).get())

        if not mraf_enabled:
            return {
                "mraf_enabled": False,
                "where_working": None,
                "signal_region": None,
                "noise_region": None,
                "zero_region": None,
            }

        noise_region = cp.isnan(self.target)

        zero_region = cp.abs(self.target) == 0
        if "zero_factor" in self.flags and self.flags["zero_factor"] != 0:
            Z = int(cp.sum(zero_region))
            if Z > 0 and not hasattr(self, "zero_weights"):
                self.zero_weights = cp.zeros((Z,), dtype=self.dtype_complex)

        signal_region = cp.logical_not(cp.logical_or(noise_region, zero_region))
        mraf_factor = self.flags.get("mraf_factor", None)
        # if mraf_factor is not None:
        #     if mraf_factor < 0:
        #         raise ValueError("mraf_factor={} should not be negative.".format(mraf_factor))

        # `where=` functionality is needed for MRAF, but this is a undocumented/new cupy feature.
        # We test whether it is available https://github.com/cupy/cupy/pull/7281
        try:
            test = cp.arange(10)
            cp.multiply(test, test, where=test > 5)
            where_working = True
        except:
            try:
                test = cp.arange(10)
                cp.multiply(test, test, _where=test > 5)
                where_working = False
            except:
                raise Exception(
                    "MRAF not supported on this system. Arithmetic `where=` is needed. "
                    "See https://github.com/cupy/cupy/pull/7281."
                )

        return {
            "mraf_enabled": mraf_enabled,
            "where_working": where_working,
            "signal_region": signal_region,
            "noise_region": noise_region,
            "zero_region": zero_region,
        }

    def _gs_farfield_routines(self, mraf_variables):
        # Weight, if desired.
        if "WGS" in self.flags["method"]:
            self._update_weights()

            # Decide whether to fix phase.
            if "Kim" in self.flags["method"]:
                was_not_fixed = not self.flags["fixed_phase"]

                # Enable based on efficiency.
                if self.flags["fix_phase_efficiency"] is not None:
                    stats = self.stats["stats"]
                    groups = tuple(stats.keys())

                    if len(stats) == 0:
                        raise ValueError(
                            "Must track statistics to fix phase based on efficiency!"
                        )

                    eff = stats[groups[-1]]["efficiency"][self.iter]
                    if eff > self.flags["fix_phase_efficiency"]:
                        self.flags["fixed_phase"] = True

                # Enable based on iterations.
                if was_not_fixed:
                    if self.iter >= self.flags["fix_phase_iteration"] - 1:
                        previous = self.stats["flags"]["fixed_phase"]
                        contiguous_falses = all(
                            [
                                not previous[-1 - i]
                                for i in range(self.flags["fix_phase_iteration"])
                            ]
                        )
                        if contiguous_falses:
                            self.flags["fixed_phase"] = True

                # Save the phase if we are going from unfixed to fixed.
                if self.flags["fixed_phase"] and self.phase_ff is None or was_not_fixed:
                    self.phase_ff = cp.arctan2(
                        self.farfield.imag, self.farfield.real, out=self.phase_ff
                    )
            else:
                self.flags["fixed_phase"] = False

        mraf_enabled = mraf_variables["mraf_enabled"]

        # Fix amplitude, potentially also fixing the phase.
        if not mraf_enabled:
            # if ("fixed_phase" in self.flags and self.flags["fixed_phase"]):
            #     # Set the farfield to the stored phase and updated weights.
            #     cp.exp(1j * self.phase_ff, out=farfield)
            #     cp.multiply(farfield, self.weights, out=farfield)
            # else:
            #     # Set the farfield amplitude to the updated weights.
            #     cp.divide(farfield, cp.abs(farfield), out=farfield)
            #     cp.multiply(farfield, self.weights, out=farfield)
            #     cp.nan_to_num(farfield, copy=False, nan=0)

            if (
                not ("fixed_phase" in self.flags and self.flags["fixed_phase"])
                or self.phase_ff is None
            ):
                self.phase_ff = cp.arctan2(
                    self.farfield.imag, self.farfield.real, out=self.phase_ff
                )

            cp.exp(1j * self.phase_ff, out=self.farfield)
            cp.multiply(self.farfield, self.weights, out=self.farfield)
        else:  # Mixed region amplitude freedom (MRAF) case.
            zero_region = mraf_variables["zero_region"]
            noise_region = mraf_variables["noise_region"]
            signal_region = mraf_variables["signal_region"]
            mraf_factor = self.flags.get("mraf_factor", None)
            where_working = mraf_variables["where_working"]

            if hasattr(self, "zero_weights"):
                fz = self.farfield[zero_region]
                self.zero_weights -= self.flags.get("zero_factor", 1) * cp.abs(fz) * fz
                self.farfield[zero_region] = self.zero_weights
            else:
                self.farfield[zero_region] = 0

            # # Handle signal and noise regions.
            # if ("fixed_phase" in self.flags and self.flags["fixed_phase"]):
            #     # Set the farfield to the stored phase and updated weights, in the signal region.
            #     if where_working:
            #         cp.exp(1j * self.phase_ff, where=signal_region, out=farfield)
            #         cp.multiply(farfield, self.weights, where=signal_region, out=farfield)
            #         if mraf_factor is not None: cp.multiply(farfield, mraf_factor, where=noise_region, out=farfield)
            #     else:
            #         cp.exp(1j * self.phase_ff, _where=signal_region, out=farfield)
            #         cp.multiply(farfield, self.weights, _where=signal_region, out=farfield)
            #         if mraf_factor is not None: cp.multiply(farfield, mraf_factor, _where=noise_region, out=farfield)
            # else:
            #     # Set the farfield amplitude to the updated weights, in the signal region.
            #     if where_working:
            #         cp.divide(farfield, cp.abs(farfield), where=signal_region, out=farfield)
            #         cp.multiply(farfield, self.weights, where=signal_region, out=farfield)
            #         if mraf_factor is not None: cp.multiply(farfield, mraf_factor, where=noise_region, out=farfield)
            #     else:
            #         cp.divide(farfield, cp.abs(farfield), _where=signal_region, out=farfield)
            #         cp.multiply(farfield, self.weights, _where=signal_region, out=farfield)
            #         if mraf_factor is not None: cp.multiply(farfield, mraf_factor, _where=noise_region, out=farfield)
            #     cp.nan_to_num(farfield, copy=False, nan=0)

            if not ("fixed_phase" in self.flags and self.flags["fixed_phase"]):
                self.phase_ff = cp.arctan2(
                    self.farfield.imag, self.farfield.real, out=self.phase_ff
                )

            if where_working:
                cp.exp(1j * self.phase_ff, where=signal_region, out=self.farfield)
                cp.multiply(
                    self.farfield, self.weights, where=signal_region, out=self.farfield
                )
                if mraf_factor is not None:
                    cp.multiply(
                        self.farfield,
                        mraf_factor,
                        where=noise_region,
                        out=self.farfield,
                    )
            else:
                cp.exp(1j * self.phase_ff, _where=signal_region, out=self.farfield)
                cp.multiply(
                    self.farfield, self.weights, _where=signal_region, out=self.farfield
                )
                if mraf_factor is not None:
                    cp.multiply(
                        self.farfield,
                        mraf_factor,
                        _where=noise_region,
                        out=self.farfield,
                    )

            # self.plot_farfield(signal_region.astype(float))
            # self.plot_farfield(noise_region.astype(float))
            # self.plot_farfield(zero_region.astype(float))
            # self.plot_farfield(np.isnan(self.farfield).astype(float))
            # self.plot_farfield(self.farfield)

        # self.farfield /= Hologram._norm(self.farfield, xp=cp)

    # Conjugate gradient optimization.
    def optimize_cg(self, iterations, callback):
        """
        Conjugate Gradient (CG) iterative phase retrieval.

        **(This feature is experimental.)**

        Solves the "phase problem": approximates the nearfield phase that
        transforms a known nearfield source amplitude to a desired farfield
        target amplitude.

        Caution
        ~~~~~~~
        This function should be called through :meth:`.optimize()` and not called
        directly. It is left as a public function exposed in documentation to clarify
        how the internals of :meth:`.optimize()` work.

        Parameters
        ----------
        iterations : iterable
            Number of loop iterations to run. Is an iterable to pass a :mod:`tqdm` iterable.
        callback : callable OR None
            See :meth:`.optimize()`.
        """
        # pytorch is optional in case some users are allergic to bloat.
        if torch is None:
            raise ValueError("pytorch is required for conjugate gradient optimization.")

        # Convert variables to torch with **zero-copy** cupy interoperability.
        # We need torch to handle gradient calculation.
        phase_torch = Hologram._get_torch_tensor_from_cupy(self.phase)
        phase_torch.requires_grad_(True)

        # Create the optimizer.
        try:
            optim_class = getattr(torch.optim, self.flags["optimizer"])
        except:
            raise ValueError(
                f"'{self.flags['optimizer']}' is not a valid torch optimizer"
            )

        self.optimizer = optim_class([phase_torch], **self.flags["optimizer_kwargs"])

        for _ in iterations:
            # (A) Step the Conjugate Gradient Optimization
            # (A.1) Reset the gradients for this step.
            self.optimizer.zero_grad()

            # (A.1) Compute the loss for this phase pattern.
            # This computes the farfield (and potentially experimental results)
            # and then passes these values to the current ``loss`` function.
            result = self._cg_loss(phase_torch)

            self.flags["loss_result"] = float(result.detach())

            if hasattr(iterations, "set_description"):
                iterations.set_description("loss=" + str(self.flags["loss_result"]))

            # (A.2) Compute the gradients of the phase pattern with respect to loss.
            result.backward(retain_graph=True)

            # (A.3) Step the optimization of phase_torch according to the gradients calculated.
            self.optimizer.step()

            # (B) Midloop Routines
            # (B.1) Run step function if present and check termination conditions.
            if callback is not None:
                if callback(self):
                    break

            # (B.2) Update statistics.
            self._update_stats(self.flags["stat_groups"])

            # Increment iteration.
            self.iter += 1

        self.phase = cp.asarray(phase_torch.detach())

        # Update the final farfield using phase and amp.
        self._populate_results()

    def _cg_loss(self, phase_torch):
        """
        Computes the loss of the current trial phase pattern.
        """
        # Grab
        farfield_torch = self._nearfield2farfield(phase_torch=phase_torch)
        target_torch = Hologram._get_torch_tensor_from_cupy(self.target)

        # Parse loss.
        loss = self.flags["loss"]
        if loss is None:
            loss = self.flags["loss"] = ComplexMSELoss()

        # Evaluate loss depending on the feedback mechanism.
        feedback = self.flags["feedback"]

        if feedback == "computational":
            return loss(farfield_torch, target_torch)
        elif feedback == "experimental":
            self.measure("knm")  # Make sure data is there.
            img_knm_torch = Hologram._get_torch_tensor_from_cupy(self.target)

            # Replace the values of the farfield with the measured values, but keep the
            # gradients using detach().
            farfield_feedback_torch = farfield_torch.detach()
            farfield_feedback_torch[:] = img_knm_torch[:]
            farfield_feedback_torch = farfield_feedback_torch.requires_grad_()

            return loss(farfield_feedback_torch, target_torch)

    @staticmethod
    def _get_torch_tensor_from_cupy(array):
        if torch is None:
            raise RuntimeError(
                "Cannot get torch tensor without torch. Something is wrong."
            )

        if array is None:
            return None
        else:
            if cp == np:
                return torch.from_numpy(array)
            else:
                return torch.as_tensor(array, device="cuda")

    # Weighting functions.
    def _update_weights_generic(
        self, weight_amp, feedback_amp, target_amp, xp=cp, nan_checks=True
    ):
        """
        Helper function to process weight feedback according to the chosen weighting method.

        Caution
        ~~~~~~~
        ``weight_amp`` **is** modified in-place.

        Parameters
        ----------
        weight_amp : numpy.ndarray OR cupy.ndarray
            Weights to update.
        feedback_amp : numpy.ndarray OR cupy.ndarray
            Measured or result amplitudes corresponding to ``weight_amp``.
            Should be the same size as ``weight_amp``.
        target_amp : numpy.ndarray OR cupy.ndarray OR None
            Necessary in the case where ``target_amp`` is not uniform, such that the weighting can
            properly be applied to bring the feedback closer to the target.
        xp : module
            This function is used by both :mod:`cupy` and :mod:`numpy`, so we have the option
            for either. Defaults to :mod:`cupy`.
        nan_checks : bool
            Whether to enable checks to avoid division by zero or ``nan`` infiltration.

        Returns
        -------
        numpy.ndarray OR cupy.ndarray
            The updated ``weight_amp``.
        """
        if self._update_weights_generic_cuda_kernel is None or xp == np:
            return self._update_weights_generic_cupy(
                weight_amp, feedback_amp, target_amp, xp, nan_checks
            )
        else:
            return self._update_weights_generic_cuda(
                weight_amp, feedback_amp, target_amp
            )

    def _update_weights_generic_cupy(
        self, weight_amp, feedback_amp, target_amp, xp=cp, nan_checks=True
    ):
        method = self.flags["method"].lower()
        if method[:4] != "wgs-":
            raise ValueError("Weighting is only for WGS.")
        method = method[4:]

        feedback_corrected = xp.array(feedback_amp, copy=True, dtype=self.dtype)
        feedback_corrected *= 1 / Hologram3D._norm(feedback_corrected, xp=xp)

        if "wu" in method or "tanh" in method:  # Additive
            feedback_corrected *= -self.flags["feedback_exponent"]
            feedback_corrected += xp.array(
                target_amp, copy=(False if np.__version__[0] == "1" else None)
            )
        else:  # Multiplicative
            xp.divide(
                feedback_corrected,
                xp.array(
                    target_amp, copy=(False if np.__version__[0] == "1" else None)
                ),
                out=feedback_corrected,
            )

            if nan_checks:
                feedback_corrected[feedback_corrected == np.inf] = 1
                feedback_corrected[
                    xp.array(
                        target_amp, copy=(False if np.__version__[0] == "1" else None)
                    )
                    == 0
                ] = 1

                xp.nan_to_num(feedback_corrected, copy=False, nan=1)

        # Fix feedback according to the desired method.
        if "leonardo" in method or "kim" in method:
            # 1/(x^p)
            xp.power(
                feedback_corrected,
                -self.flags["feedback_exponent"],
                out=feedback_corrected,
            )
        elif "nogrette" in method.lower():
            # Taylor expand 1/(1-g(1-x)) -> 1 + g(1-x) + (g(1-x))^2 ~ 1 + g(1-x)
            feedback_corrected *= -(1 / xp.nanmean(feedback_corrected))
            feedback_corrected += 1
            feedback_corrected *= -self.flags["feedback_factor"]
            feedback_corrected += 1
            xp.reciprocal(feedback_corrected, out=feedback_corrected)
        elif "wu" in method:
            feedback_corrected = np.exp(
                self.flags["feedback_exponent"] * feedback_corrected
            )
        elif "tanh" in method:
            feedback_corrected = self.flags["feedback_factor"] * np.tanh(
                self.flags["feedback_exponent"] * feedback_corrected
            )
            feedback_corrected += 1
        else:
            raise ValueError(
                f"Method '{self.flags['method']}' not recognized by Hologram.optimize()"
            )

        if nan_checks:
            feedback_corrected[feedback_corrected == np.inf] = 1

        # Update the weights.
        weight_amp *= feedback_corrected

        if nan_checks:
            xp.nan_to_num(weight_amp, copy=False, nan=0.0001)
            # weight_amp[weight_amp == np.inf] = 1

        # Normalize amp, as methods may have broken conservation.
        weight_amp *= 1 / Hologram._norm(weight_amp, xp=xp)

        return weight_amp

    def _update_weights_generic_cuda(self, weight_amp, feedback_amp, target_amp):
        N = weight_amp.size

        feedback_amp = cp.array(
            feedback_amp, copy=(False if np.__version__[0] == "1" else None)
        )
        feedback_norm = Hologram._norm(feedback_amp, xp=cp)

        method = ALGORITHM_INDEX[self.flags["method"]]

        threads_per_block = int(
            self._update_weights_generic_cuda_kernel.max_threads_per_block
        )
        blocks = N // threads_per_block + 1

        # Call the RawKernel.
        self._update_weights_generic_cuda_kernel(
            (blocks,),
            (threads_per_block,),
            (
                weight_amp,
                feedback_amp,
                target_amp,
                N,
                method,
                feedback_norm,
                self.flags.pop("feedback_exponent", 1),
                self.flags.pop("feedback_factor", 1),
            ),
        )

        weight_amp *= 1 / Hologram._norm(weight_amp, xp=cp)

        return weight_amp

    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude.
        """
        feedback = self.flags["feedback"]

        if feedback == "computational":
            self._update_weights_generic(self.weights, self.amp_ff, self.target)

    # Other helper functions.
    @staticmethod
    def set_mempool_limit(device=0, size=None, fraction=None):
        """
        Helper function to set the `cupy memory pool size
        <https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPool.html#cupy.cuda.MemoryPool>`_.

        Parameters
        ----------
        device : int
            Which GPU to set the limit on. Passed to :meth:`cupy.cuda.Device()`.
        size : int
            Desired number of bytes in the pool. Passed to :meth:`cupy.cuda.MemoryPool.set_limit()`.
        fraction : float
            Fraction of available memory to use. Passed to :meth:`cupy.cuda.MemoryPool.set_limit()`.
        """
        if cp == np:
            raise ValueError("Cannot set mempool for numpy. Need cupy.")

        mempool = cp.get_default_memory_pool()

        with cp.cuda.Device(device):
            mempool.set_limit(size=size, fraction=fraction)

            print(
                "cupy memory pool limit set to {:.2f} GB...".format(
                    mempool.get_limit() / (1024.0**3)
                )
            )

    @staticmethod
    def get_mempool_limit(device=0):
        """
        Helper function to get the `cupy memory pool size
        <https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPool.html#cupy.cuda.MemoryPool>`_.

        Parameters
        ----------
        device : int
            Which GPU to set the limit on. Passed to :meth:`cupy.cuda.Device()`.

        Returns
        -------
        int
            Current memory pool limit in bytes
        """

        if cp == np:
            raise ValueError("Cannot get mempool for numpy. Need cupy.")

        mempool = cp.get_default_memory_pool()

        with cp.cuda.Device(device):
            return mempool.get_limit()

    @staticmethod
    def _norm(matrix, xp=cp):
        r"""
        Computes the root of the sum of squares of the given ``matrix``. Implements

        .. math:: \sqrt{\iint |\vec{E}|^2}.

        Parameters
        ----------
        matrix : numpy.ndarray OR cupy.ndarray
            Data, potentially complex.
        xp : module
            This function is used by :mod:`cupy`, :mod:`numpy`, and :mod:`torch`,
            so we have the option for any of the three. Defaults to :mod:`cupy`.

        Returns
        -------
        float
            The result.
        """
        if torch is not None:
            if torch.is_tensor(matrix):
                xp = torch

        if xp is torch:
            is_complex = torch.is_complex(matrix)
        else:
            is_complex = xp.iscomplexobj(matrix)

        if is_complex:
            return xp.sqrt(xp.nansum(xp.square(xp.abs(matrix))))
        else:
            return xp.sqrt(xp.nansum(xp.square(matrix)))

    def plot_farfield(
        self,
        source=None,
        title="",
        limits=None,
        units="knm",
        limit_padding=0.1,
        figsize=(12, 6),
        cbar=False,
        animate_3d=False,
        frames=100,
        interval=50,
    ):
        """
        Plots an overview (left) and zoom (right) view of ``source``, with an optional 3D animated plot.

        Parameters
        ----------
        source : array_like OR None
            Should have shape equal to :attr:`shape`.
            If ``None``, defaults to :attr:`amp_ff`.
        title : str
            Title of the plots. If ``"phase"`` is a substring of title, then the source
            is treated as a phase.
        limits : ((float, float), (float, float), (float, float)) OR None
            :math:`x`, :math:`y`, and :math:`z` limits for the zoom plot in ``"knm"`` space.
        units : str
            Far-field units for plots.
        limit_padding : float
            Fraction of the width and height to expand the limits of the zoom plot by.
        figsize : tuple
            Size of the plot.
        cbar : bool
            Whether to add colorbars to the plots.
        animate_3d : bool
            Whether to animate the 3D plot with sinusoidal motion along the z-axis.
        frames : int
            Number of frames for the animation.
        interval : int
            Time interval between frames in milliseconds.

        Returns
        -------
        ((float, float), (float, float), (float, float))
            Used ``limits``, which may be autocomputed. Autocomputed limits are returned
            as integers.
        """
        # Parse source
        if source is None:
            source = (
                self.amp_ff if self.amp_ff is not None else self.get_farfield(get=False)
            )

        # Interpret source and convert to numpy for plotting
        isphase = "phase" in title.lower()
        npsource = (
            np.mod(source.get() if hasattr(source, "get") else source, 2 * np.pi)
            if isphase
            else np.abs(source.get() if hasattr(source, "get") else source)
        )

        # Compute limits if not provided
        if limits is None:
            limits = self._compute_limits(npsource, limit_padding=limit_padding)

        # Create figure and subplots
        fig = plt.figure(figsize=figsize)
        axs_2d = fig.add_subplot(1, 3 if animate_3d else 2, 1)
        axs_zoom = fig.add_subplot(1, 3 if animate_3d else 2, 2)

        # Full field plot
        axs_2d.imshow(
            npsource,
            cmap=("twilight" if isphase else None),
            interpolation=("none" if isphase else "gaussian"),
        )
        axs_2d.set_title(f"{title} Full")

        # Zoom plot
        zoom_data = npsource[
            np.ix_(
                np.arange(limits[1][0], limits[1][1]),
                np.arange(limits[0][0], limits[0][1]),
            )
        ]
        axs_zoom.imshow(
            zoom_data,
            cmap=("twilight" if isphase else None),
            extent=[limits[0][0], limits[0][1], limits[1][1], limits[1][0]],
            interpolation="none",
        )
        axs_zoom.set_title(f"{title} Zoom")

        # 3D Animation
        if animate_3d:
            ax_3d = fig.add_subplot(1, 3, 3, projection="3d")
            x = np.linspace(0, npsource.shape[1], npsource.shape[1])
            y = np.linspace(0, npsource.shape[0], npsource.shape[0])
            X, Y = np.meshgrid(x, y)

            # Initialize the 3D surface
            Z = npsource
            surf = ax_3d.plot_surface(
                X, Y, Z, cmap="viridis" if not isphase else "twilight", edgecolor="none"
            )
            ax_3d.set_zlim(0, np.nanmax(npsource) * 1.5)
            ax_3d.set_title(f"{title} 3D View")
            ax_3d.set_xlabel("X axis")
            ax_3d.set_ylabel("Y axis")
            ax_3d.set_zlabel("Amplitude")

            # Update function for animation
            def update(frame):
                ax_3d.clear()
                Z = npsource * np.sin(2 * np.pi * frame / frames)
                ax_3d.plot_surface(
                    X,
                    Y,
                    Z,
                    cmap="viridis" if not isphase else "twilight",
                    edgecolor="none",
                )
                ax_3d.set_zlim(0, np.nanmax(npsource) * 1.5)
                ax_3d.set_title(f"{title} 3D View - Frame {frame}")
                ax_3d.set_xlabel("X axis")
                ax_3d.set_ylabel("Y axis")
                ax_3d.set_zlabel("Amplitude")

            # Create the animation
            ani = FuncAnimation(
                fig, update, frames=frames, interval=interval, repeat=True
            )

        # Show colorbar if requested
        if cbar:
            fig.colorbar(axs_zoom.images[0], ax=axs_zoom, orientation="vertical")

        plt.tight_layout()
        plt.show()

        return limits


class FeedbackHologram3D(Hologram3D):
    """
    Experimental holography aided by camera feedback.
    Contains mechanisms for hologram positioning and camera feedback aided by a
    :class:`~slmsuite.hardware.cameraslms.FourierSLM`.

    Attributes
    ----------
    cameraslm : slmsuite.hardware.cameraslms.FourierSLM OR None OR (int, int)
        A hologram with experimental feedback needs access to an SLM and camera.
        If ``None``, no feedback is applied (mostly defaults to :class:`Hologram`).
    _cam_points : numpy.ndarray
        Array containing points corresponding to the corners of the camera in the SLM's
        k-space. At the moment, this is only used for plotting.
        First point is repeated at the end.
    target_ij :  array_like OR None
        Amplitude target in the ``"ij"`` (camera) basis. Of same ``shape`` as the camera in
        :attr:`cameraslm`.  Counterpart to :attr:`target` which is in the ``"knm"``
        (computational k-space) basis.
    img_ij, img_knm
        Cached **amplitude** feedback image in the
        ``"ij"`` (raw camera) basis or
        ``"knm"`` (transformed to computational k-space) basis.
        Measured with :meth:`.measure()`.
    """

    def __init__(
        self,
        shape,
        target_ij=None,
        cameraslm=None,
        null_region=None,
        null_region_radius_frac=None,
        **kwargs,
    ):
        """
        Initializes a hologram with camera feedback.

        Parameters
        ----------
        shape : (int, int)
            Computational shape of the SLM in :mod:`numpy` `(h, w)` form. See :meth:`.Hologram.__init__()`.
        target_ij : array_like OR None
            See :attr:`target_ij`.
            Is ``None`` only if the :attr:`target` will be generated by other means
            (used by subclass :class:`SpotHologram`). This should not generally be used.

            Note
            ~~~~
            There is not currently a way to request a target in the ``"knm"`` basis and
            use the camera for feedback. In particular, the analog ``knmslm_to_ijcam``
            for :meth:`ijcam_to_knmslm()` is not written, but is definitely possible.
        cameraslm : slmsuite.hardware.cameraslms.FourierSLM OR slmsuite.hardware.slms.SLM OR None
            Provides access to experimental feedback.
            If an :class:`~slmsuite.hardware.slms.SLM` is passed, the attribute is set to ``None``,
            but the information contained in the SLM is passed to the superclass :class:`.Hologram`.
            See :attr:`cameraslm`.
        null_region : array_like OR None
            Array of shape :attr:`shape`. Where ``True``, sets the background to zero
            instead of ``nan``. If ``None``, has no effect.
        null_region_radius_frac : float OR None
            Helper function to set the ``null_region`` to zero for Fourier space radius fractions above
            ``null_region_radius_frac``. This is useful to prevent power being deflected
            to very high orders, which are unlikely to be properly represented in
            practice on a physical SLM.
        **kwargs
            Passed to :meth:`Hologram.__init__`.
        """
        # Use the Hologram constructor to initialize self.target with proper shape,
        # pass other arguments (esp. slm_shape).
        self.cameraslm = cameraslm
        if self.cameraslm is not None:
            # Determine camera size in SLM-space.
            try:
                amp = self.cameraslm.slm._get_source_amplitude()
                slm_shape = self.cameraslm.slm.shape
            except:
                # See if an SLM was passed.
                try:
                    amp = self.cameraslm._get_source_amplitude()
                    slm_shape = self.cameraslm.shape

                    # We don't have access to all the calibration stuff, so don't
                    # confuse the rest of the init/etc.
                    self.cameraslm = None
                except:
                    raise ValueError(
                        "Expected a CameraSLM or SLM to be passed to cameraslm."
                    )

        else:
            amp = kwargs.pop("amp", None)
            slm_shape = None

        if not "slm_shape" in kwargs:
            kwargs["slm_shape"] = slm_shape

        super().__init__(target=shape, amp=amp, **kwargs)

        self.img_ij = None
        self.img_knm = None
        if target_ij is None:
            self.target_ij = None
        else:
            self.target_ij = target_ij.astype(self.dtype)

        if self.cameraslm is not None and "fourier" in self.cameraslm.calibrations:
            # Generate a list of the corners of the camera, for plotting.
            cam_shape = self.cameraslm.cam.shape

            ll = [0, 0]
            lr = [0, cam_shape[0] - 1]
            ur = [cam_shape[1] - 1, cam_shape[0] - 1]
            ul = [cam_shape[1] - 1, 0]

            points_ij = toolbox.format_2vectors(np.vstack((ll, lr, ur, ul, ll)).T)
            points_kxy = self.cameraslm.ijcam_to_kxyslm(points_ij)
            self._cam_points = toolbox.convert_vector(
                points_kxy,
                from_units="kxy",
                to_units="knm",
                hardware=self.cameraslm.slm,
                shape=self.shape,
            )

            # Transform the target, if it is provided.
            if target_ij is not None:
                self.update_target(
                    target_ij, null_region, null_region_radius_frac, reset_weights=True
                )

        else:
            self._cam_points = None

    # Image transformation helper function.
    def ijcam_to_knmslm(self, img, out=None, blur_ij=None, order=3):
        """
        Convert an image in the camera domain to computational SLM k-space using, in part, the
        affine transformation stored in a cameraslm's Fourier calibration.

        Note
        ~~~~
        This includes two transformations:

        - The affine transformation ``"ij"`` -> ``"kxy"`` (camera pixels to normalized k-space).
        - The scaling ``"kxy"`` -> ``"knm"`` (normalized k-space to computational k-space pixels).

        Parameters
        ----------
        img : numpy.ndarray OR cupy.ndarray
            Image to transform. This should be the same shape as images returned by the camera.
        out : numpy.ndarray OR cupy.ndarray OR None
            If ``out`` is not ``None``, this array will be used to write the memory in-place.
        blur_ij : int OR None
            Applies a ``blur_ij`` pixel-width Gaussian blur to ``img``.
            If ``None``, defaults to the ``"blur_ij"`` flag if present; otherwise zero.
        order : int
            Order of interpolation used for transformation. Defaults to 3 (cubic).

        Returns
        -------
        numpy.ndarray OR cupy.ndarray
            Image transformed into ``"knm"`` space.
        """
        if self.cameraslm is None:
            raise RuntimeError(
                "Cannot use ijcam_to_knmslm without the calibrations in a cameraslm."
            )
        if not "fourier" in self.cameraslm.calibrations:
            raise RuntimeError("ijcam_to_knmslm requires a Fourier calibration.")

        # First transformation. FUTURE: make convert_basis to output a matrix like here?
        conversion = toolbox.convert_vector(
            (1, 1), "knm", "kxy", hardware=self.cameraslm.slm, shape=self.shape
        ) - toolbox.convert_vector(
            (0, 0), "knm", "kxy", hardware=self.cameraslm.slm, shape=self.shape
        )
        M1 = np.diag(np.squeeze(conversion))
        b1 = np.matmul(
            M1, -toolbox.format_2vectors(np.flip(np.squeeze(self.shape)) / 2)
        )

        # Second transformation.
        M2 = self.cameraslm.calibrations["fourier"]["M"]
        b2 = self.cameraslm.calibrations["fourier"]["b"]
        if "a" in self.cameraslm.calibrations["fourier"]:
            b2 -= np.matmul(M2, self.cameraslm.calibrations["fourier"]["a"])

        # Composite transformation (along with xy -> yx).
        M = cp.array(np.flip(np.flip(np.matmul(M2, M1), axis=0), axis=1))
        b = cp.array(np.flip(np.squeeze(np.matmul(M2, b1) + b2)))

        # See if the user wants to blur.
        if blur_ij is None:
            if "blur_ij" in self.flags:
                blur_ij = self.flags["blur_ij"]
            else:
                blur_ij = 0

        # FUTURE: use cp_gaussian_filter (faster?); was having trouble with cp_gaussian_filter.
        if blur_ij > 0:
            img = sp_gaussian_filter(img, (blur_ij, blur_ij), output=img, truncate=2)

        cp_img = cp.array(img, dtype=self.dtype)
        cp.abs(cp_img, out=cp_img)

        # Perform affine.
        target = cp_affine_transform(
            input=cp_img,
            matrix=M,
            offset=b,
            output_shape=self.shape,
            order=order,
            output=out,
            mode="constant",
            cval=np.nan,
        )

        # Filter the image. FUTURE: fix.
        # target = cp_gaussian_filter1d(target, blur, axis=0, output=target, truncate=2)
        # target = cp_gaussian_filter1d(target, blur, axis=1, output=target, truncate=2)

        target = cp.abs(target, out=target)
        norm = Hologram._norm(target)
        target *= 1 / norm

        if norm == 0:
            raise ValueError(
                "No power in hologram. Maybe target_ij is out of range of knm space? "
                "Check transformations."
            )

        return target

    # Measurement.
    def measure(self, basis="ij"):
        """
        Method to request a measurement to occur. If :attr:`img_ij` is ``None``,
        then a new image will be grabbed from the camera (this is done automatically in
        algorithms).

        Parameters
        ----------
        basis : str
            The cached image to be sure to fill with new data.
            Can be ``"ij"`` or ``"knm"``.

            - If ``"knm"``, then :attr:`img_ij` and :attr:`img_knm` are filled.
            - If ``"ij"``, then :attr:`img_ij` is filled, and :attr:`img_knm` is ignored.

            This is useful to avoid (expensive) transformation from the ``"ij"`` to the
            ``"knm"`` basis if :attr:`img_knm` is not needed.
        """
        if self.img_ij is None and (basis == "knm" or basis == "ij"):
            # Apply the pattern to the SLM at the desired depth (implemented by propagation_kernel)
            self.cameraslm.slm.set_phase(
                self.get_phase(include_propagation=True), settle=True
            )

            # Measure the result.
            self.cameraslm.cam.flush()
            self.img_ij = np.array(
                self.cameraslm.cam.get_image(),
                copy=(False if np.__version__[0] == "1" else None),
                dtype=self.dtype,
            )

            if basis == "knm":  # Compute the knm basis image.
                self.img_knm = self.ijcam_to_knmslm(self.img_ij, out=self.img_knm)
                cp.sqrt(self.img_knm, out=self.img_knm)
            else:  # The old image is outdated, erase it. FUTURE: memory concerns?
                self.img_knm = None

            self.img_ij = np.sqrt(
                self.img_ij
            )  # Don't load to the GPU if not necessary.
        elif basis == "knm":
            if self.img_knm is None:
                self.img_knm = self.ijcam_to_knmslm(
                    np.square(self.img_ij), out=self.img_knm
                )
                cp.sqrt(self.img_knm, out=self.img_knm)
        elif basis == "ij":
            pass
        else:
            raise ValueError(
                f"Unrecognized measurement basis '{basis}'. Options are 'ij' or 'knm'"
            )

    # Target update.
    def update_target(
        self,
        new_target_ij,
        null_region=None,
        null_region_radius_frac=None,
        reset_weights=False,
    ):
        """
        Change the target to something new. This method handles cleaning and normalization.

        Parameters
        ----------
        new_target_ij : array_like OR None
            New :attr:`target_ij` to optimize towards *in the camera basis*.
            should be of the same shape as the camera.
            Also updates :attr:`target` using the stored Fourier calibration.
        null_region : array_like OR None
            Array of shape :attr:`shape`. Where ``True``, sets the background to zero
            instead of ``nan``. If ``None``, has no effect.
        null_region_radius_frac : float OR None
            Helper function to set the ``null_region`` to zero for Fourier space radius fractions above
            ``null_region_radius_frac``. This is useful to prevent power being deflected
            to very high orders, which are unlikely to be properly represented in
            practice on a physical SLM. If ``None``, defaults to 1 and there is no null region.
        reset_weights : bool
            Whether to update the :attr:`weights` to this new :attr:`target`.
        """
        self.target_ij = new_target_ij.astype(self.dtype)
        # Transformation order of zero to prevent nan-blurring in MRAF cases.
        self.ijcam_to_knmslm(new_target_ij, out=self.target, order=0)

        # Set the null region.
        undefined = cp.isnan(self.target)

        if null_region_radius_frac is None:
            null_region_radius_frac = 1

        if null_region_radius_frac < 1:
            # Build up the null region pattern if we have not already done the transform above.
            if null_region is None:
                null_region = cp.zeros(self.shape, dtype=bool)

            # Make a circle, outside of which the null_region is active.
            xl = cp.linspace(-1, 1, null_region.shape[0])
            yl = cp.linspace(-1, 1, null_region.shape[1])
            (xg, yg) = cp.meshgrid(xl, yl)
            mask = cp.square(xg) + cp.square(yg) > null_region_radius_frac**2
            null_region[mask] = True

        if null_region_radius_frac >= 1:
            self.target[undefined] = 0
        else:
            self.target[cp.logical_and(undefined, null_region)] = 0

        if reset_weights:
            self.reset_weights()

    def refine_offset(self, img, basis="kxy"):
        """
        **(NotImplemented)**
        Hones the position of the produced image to the desired target image to compensate for
        Fourier calibration imperfections. Works either by moving the desired camera
        target to align where the image ended up (``basis="ij"``) or by moving
        the :math:`k`-space image to target the desired camera target
        (``basis="knm"``/``basis="kxy"``). This should be run at the user's request
        inbetween :meth:`optimize` iterations.

        Parameters
        ----------
        img : numpy.ndarray
            Image measured by the camera.
        basis : str
            The correction can be in any of the following bases:
            - ``"ij"`` changes the pixel that the spot is expected at,
            - ``"kxy"`` or ``"knm"`` changes the k-vector which the SLM targets.
            Defaults to ``"kxy"`` if ``None``.

        Returns
        -------
        numpy.ndarray
            Euclidean pixel error in the ``"ij"`` basis for each spot.
        """
        # Probably local autocorrelation algorithm.

        raise NotImplementedError()

    # Weighting and stats.
    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude.
        """
        feedback = self.flags["feedback"]

        if feedback == "computational":
            self._update_weights_generic(self.weights, self.amp_ff, self.target)
        elif feedback == "experimental":
            self.measure("knm")  # Make sure data is there.
            self._update_weights_generic(self.weights, self.img_knm, self.target)

    def _calculate_stats_experimental(self, stats, stat_groups=[]):
        """
        Wrapped by :meth:`FeedbackHologram._update_stats()`.
        """
        if "experimental_knm" in stat_groups:
            self.measure("knm")  # Make sure data is there.

            stats["experimental_knm"] = self._calculate_stats(
                self.img_knm,
                self.target,
                efficiency_compensation=True,
                raw="raw_stats" in self.flags and self.flags["raw_stats"],
            )
        if "experimental_ij" in stat_groups or "experimental" in stat_groups:
            self.measure("ij")  # Make sure data is there.

            stats["experimental_ij"] = self._calculate_stats(
                self.img_ij,
                self.target_ij,
                xp=np,
                efficiency_compensation=True,
                raw="raw_stats" in self.flags and self.flags["raw_stats"],
            )

    def _update_stats(self, stat_groups=[]):
        """
        Calculate statistics corresponding to the desired ``stat_groups``.

        Parameters
        ----------
        stat_groups : list of str
            Which groups or types of statistics to analyze.
        """
        stats = {}

        self._calculate_stats_computational(stats, stat_groups)
        self._calculate_stats_experimental(stats, stat_groups)

        self._update_stats_dictionary(stats)


class _Abstract3DSpotHologram(FeedbackHologram3D):
    """
    Abstract class to eventally handle :meth:`SpotHologram.refine_offset()`
    and other shared methods for :class:`SpotHologram` and :class:`CompressedSpotHologram`.
    There are many parts of :class:`SpotHologram` with repetition and bloat that
    can be simplified with more modern features from other parts of :mod:`slmsuite`.
    """

    pass

    def remove_vortices(self):
        """Spot holograms do not need to consider vortices."""
        pass

    # def update_spots(self, source_basis="kxy"):
    #     pass


class SpotHologram3D(_Abstract3DSpotHologram):
    """
    Holography optimized for the generation of three dimensional optical focal arrays (DFT-based).

    Is a subclass of :class:`FeedbackHologram`, but falls back to non-camera-feedback
    routines if :attr:`cameraslm` is not passed.

    Tip
    ~~~
    Quality of life features to generate noise regions for mixed region amplitude
    freedom (MRAF) algorithms are supported. Specifically, set ``null_region``
    parameters to help specify where the noise region is not.

    Attributes
    ----------
    spot_knm, spot_kxy, spot_ij : array_like of float OR None
        Stored vectors with shape ``(3, N)`` in the style of
        :meth:`~slmsuite.holography.toolbox.format_vectors()`.
        These vectors are floats.
        The subscript refers to the basis of the vectors, the transformations between
        which are autocomputed.
        If necessary transformations do not exist, :attr:`spot_ij` is set to ``None``.
    spot_knm_rounded : array_like of int
        :attr:`spot_knm` rounded to nearest integers (indices).
        These vectors are integers.
        This is necessary because
        GS algorithms operate on a pixel grid, and the target for each spot in a
        :class:`SpotHologram` is a single pixel (index).
    spot_kxy_rounded, spot_ij_rounded : array_like of float
        Once :attr:`spot_knm_rounded` is rounded, the original :attr:`spot_kxy`
        and :attr:`spot_ij` are no longer accurate. Transformations are again used
        to backcompute the positions in the ``"ij"`` and ``"kxy"`` bases corresponding
        to the true computational location of a given spot.
        These vectors are floats.
    spot_amp : array_like of float
        The target amplitude for each spot.
        Must have length corresponding to the number of spots.
        For instance, the user can request dimmer or brighter spots.
    external_spot_amp : array_like of float
        When using ``"external_spot"`` feedback or the ``"external_spot"`` stat group,
        the user must supply external data. This data is transferred through this
        attribute. For iterative feedback, have the ``callback()`` function set
        :attr:`external_spot_amp` dynamically. By default, this variable is set to even
        distribution of amplitude.
    spot_integration_width_knm : int
        For spot-specific feedback methods, better SNR is achieved when integrating over
        many farfield pixels. This variable stores the width of the integration region
        in ``"knm"`` (farfield) space.
    spot_integration_width_ij : int
        For spot-specific feedback methods, better SNR is achieved when integrating over
        many camera pixels. This variable stores the width of the integration region
        in ``"ij"`` (camera) space.
    null_knm : array_like of float OR None
        In addition to points where power is desired, :class:`SpotHologram` is equipped
        with quality of life features to select points where power is undesired. These
        points are stored in :attr:`null_knm` with shape ``(2, M)`` in the style of
        :meth:`~slmsuite.holography.toolbox.format_2vectors()`. A region around these
        points is set to zero (null) and not allowed to participate in the noise region.
    null_radius_knm : float
        The radius in ``"knm"`` space around the points :attr:`null_knm` to zero or null
        (prevent from participating in the ``nan`` noise region).
        This is useful to prevent power being deflected to very high orders,
        which are unlikely to be properly represented in practice on a physical SLM.
    null_region_knm : array_like of bool OR ``None``
        Array of shape :attr:`shape`. Where ``True``, sets the background to zero
        instead of nan. If ``None``, has no effect.
    """

    def __init__(
        self,
        shape,
        spot_vectors,
        basis="kxy",
        spot_amp=None,
        cameraslm=None,
        null_vectors=None,
        null_radius=None,
        null_region=None,
        null_region_radius_frac=None,
        **kwargs,
    ):
        """
        Initializes a :class:`SpotHologram` targeting given spots at ``spot_vectors``.

        Parameters
        ----------
        shape : (int, int)
            Computational shape of the SLM. See :meth:`.Hologram.__init__()`.
        spot_vectors : array_like
            Spot position vectors with shape ``(2, N)`` in the style of
            :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
        basis : str
            The spots can be in any of the following bases:

            - ``"kxy"`` for centered normalized SLM :math:`k`-space (radians),
            - ``"knm"`` for computational SLM :math:`k`-space (pixels),
            - ``"ij"`` for camera coordinates (pixels).

            Defaults to ``"kxy"``.
        spot_amp : array_like OR None
            The amplitude to target for each spot.
            See :attr:`SpotHologram.spot_amp`.
            If ``None``, all spots are assumed to have the same amplitude.
            Normalization is performed automatically; the user is not required to
            normalize.
        cameraslm : slmsuite.hardware.cameraslms.FourierSLM OR None
            If the ``"ij"`` or ``"kxy"`` bases are chosen, and/or if the user wants to make use of camera
            feedback, a :class:`slmsuite.hardware.cameraslms.FourierSLM` must be provided.
        null_vectors : array_like OR None
            Null position vectors with shape ``(2, N)`` in the style of
            :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
            MRAF methods are forced zero around these points.
        null_radius : float OR None
            Radius to null around in the given ``basis``.
            Note that basis conversions are imperfect for anisotropic basis
            transformations. The radius will always be set to be circular in ``"knm"``
            space, and will attempt to match to the closest circle
            to the (potentially elliptical) projection into ``"knm"`` from the given ``basis``.
        null_region : array_like OR None
            Array of shape :attr:`shape`. Where ``True``, sets the background to zero
            instead of ``nan``. If ``None``, has no effect.
        null_region_radius_frac : float OR None
            Helper function to set the ``null_region`` to zero for Fourier space radius fractions above
            ``null_region_radius_frac``. This is useful to prevent power being deflected
            to very high orders, which are unlikely to be properly represented in
            practice on a physical SLM.
        **kwargs
            Passed to :meth:`.FeedbackHologram.__init__()`.
        """
        # Parse vectors.
        vectors = toolbox.format_vectors(spot_vectors, expected_dimension=3)
        N = vectors.shape[1]

        # Parse spot_amp.
        if spot_amp is not None:
            self.spot_amp = np.ravel(spot_amp)
            if len(self.spot_amp) != N:
                raise ValueError(
                    "spot_amp must have the same length as the provided spots."
                )
        else:
            self.spot_amp = np.full(N, 1.0 / np.sqrt(N))

        # Set the external amp variable to be perfect by default.
        self.external_spot_amp = np.copy(self.spot_amp)

        # Parse null_vectors
        if null_vectors is not None:
            null_vectors = toolbox.format_vectors(null_vectors, expected_dimension=3)
            if not np.all(np.shape(null_vectors) == np.shape(null_vectors)):
                raise ValueError(
                    "null_vectors must have the same length as the provided spots."
                )
        else:
            self.null_knm = None
            self.null_radius_knm = None
        self.null_region_knm = None

        # Interpret vectors depending upon the basis.
        if basis is None or basis == "knm":  # Computational Fourier space of SLM.
            self.spot_knm = vectors

            if cameraslm is not None:
                self.spot_kxy = toolbox.convert_vector(
                    self.spot_knm,
                    from_units="knm",
                    to_units="kxy",
                    hardware=cameraslm,
                    shape=shape,
                )

                if "fourier" in cameraslm.calibrations:
                    self.spot_ij = cameraslm.kxyslm_to_ijcam(self.spot_kxy)
                else:
                    self.spot_ij = None
            else:
                self.spot_kxy = None
                self.spot_ij = None

            # Handle null parameters.
            self.null_knm = null_vectors
            self.null_radius_knm = null_radius
            self.null_region_knm = null_region
        elif basis == "kxy":  # Normalized units.
            assert cameraslm is not None, "We need a cameraslm to interpret kxy."

            self.spot_kxy = vectors

            if hasattr(cameraslm, "calibrations"):
                if "fourier" in cameraslm.calibrations:
                    self.spot_ij = cameraslm.kxyslm_to_ijcam(vectors)
                    # This is okay for non-feedback GS, so we don't error.
            else:
                self.spot_ij = None

            self.spot_knm = toolbox.convert_vector(
                self.spot_kxy,
                from_units="kxy",
                to_units="knm",
                hardware=cameraslm,
                shape=shape,
            )
        elif basis == "ij":  # Pixel on the camera.
            assert cameraslm is not None, "We need an cameraslm to interpret ij."
            assert "fourier" in cameraslm.calibrations, (
                "We need an cameraslm with "
                "fourier-calibrated kxyslm_to_ijcam and ijcam_to_kxyslm transforms "
                "to interpret ij."
            )

            self.spot_ij = vectors
            self.spot_kxy = cameraslm.ijcam_to_kxyslm(vectors)
            self.spot_knm = toolbox.convert_vector(
                vectors,
                from_units="ij",
                to_units="knm",
                hardware=cameraslm,
                shape=shape,
            )
        else:
            raise Exception("Unrecognized basis for spots '{}'.".format(basis))

        # Handle null conversions in the ij or kxy cases.
        if basis == "ij" or basis == "kxy":
            if null_vectors is not None:
                # Convert the null vectors.
                self.null_knm = toolbox.convert_vector(
                    null_vectors,
                    from_units=basis,
                    to_units="knm",
                    hardware=cameraslm,
                    shape=shape,
                )

                # Convert the null radius.
                if null_radius is not None:
                    self.null_radius_knm = toolbox.convert_radius(
                        null_radius,
                        from_units=basis,
                        to_units="knm",
                        hardware=cameraslm,
                        shape=shape,
                    )
                else:
                    self.null_radius_knm = None
            else:
                self.null_knm = None
                self.null_radius_knm = None

            self.null_region_knm = null_region

        # Generate point spread functions (psf) for the knm and ij bases
        if cameraslm is not None:
            psf_kxy = np.mean(cameraslm.slm.get_spot_radius_kxy())
            psf_knm = toolbox.convert_radius(
                psf_kxy, "kxy", "knm", cameraslm.slm, shape
            )
            psf_ij = toolbox.convert_radius(psf_kxy, "kxy", "ij", cameraslm, shape)
        else:
            psf_knm = 0
            psf_ij = np.nan

        if np.isnan(psf_knm):
            psf_knm = 0
        if np.isnan(psf_ij):
            psf_ij = 0

        # Use semi-arbitrary values to determine integration widths. The default width is:
        #  - N times the psf,
        #  - but then clipped to be:
        #    + larger than 3 and
        #    + smaller than the minimum inf-norm distance between spots divided by 1.5
        #      (divided by 1 would correspond to the largest non-overlapping integration
        #      regions; 1.5 gives comfortable padding)
        #  - and finally forced to be an odd integer.
        N = 10  # Future: non-arbitrary
        min_psf = 3

        dist_knm = np.max([toolbox.smallest_3D_distance(self.spot_knm) / 1.5, min_psf])
        self.spot_integration_width_knm = np.clip(N * psf_knm, min_psf, dist_knm)
        self.spot_integration_width_knm = int(
            2 * np.floor(self.spot_integration_width_knm / 2) + 1
        )

        if self.spot_ij is not None:
            dist_ij = np.max(
                [toolbox.smallest_3D_distance(self.spot_ij) / 1.5, min_psf]
            )
            self.spot_integration_width_ij = np.clip(N * psf_ij, min_psf, dist_ij)
            self.spot_integration_width_ij = int(
                2 * np.floor(self.spot_integration_width_ij / 2) + 1
            )
        else:
            self.spot_integration_width_ij = None

        # Check to make sure spots are within relevant camera and SLM shapes.
        if (
            np.any(self.spot_knm[0] < self.spot_integration_width_knm / 2)
            or np.any(self.spot_knm[1] < self.spot_integration_width_knm / 2)
            or np.any(
                self.spot_knm[0] >= shape[1] - self.spot_integration_width_knm / 2
            )
            or np.any(
                self.spot_knm[1] >= shape[0] - self.spot_integration_width_knm / 2
            )
        ):
            raise ValueError(
                "Spots outside SLM computational space bounds!\nSpots:\n{}\nBounds: {}".format(
                    self.spot_knm, shape
                )
            )

        if self.spot_ij is not None:
            cam_shape = cameraslm.cam.shape

            if (
                np.any(self.spot_ij[0] < self.spot_integration_width_ij / 2)
                or np.any(self.spot_ij[1] < self.spot_integration_width_ij / 2)
                or np.any(
                    self.spot_ij[0] >= cam_shape[1] - self.spot_integration_width_ij / 2
                )
                or np.any(
                    self.spot_ij[1] >= cam_shape[0] - self.spot_integration_width_ij / 2
                )
            ):
                raise ValueError(
                    "Spots outside camera bounds!\nSpots:\n{}\nBounds: {}".format(
                        self.spot_ij, cam_shape
                    )
                )

        # Decide the null_radius (if necessary)
        if self.null_knm is not None:
            if self.null_radius_knm is None:
                all_spots = np.hstack((self.null_knm, self.spot_knm))
                self.null_radius_knm = toolbox.smallest_distance(all_spots) / 4

            self.null_radius_knm = int(np.ceil(self.null_radius_knm))

        # Initialize target/etc. Note that this passes through FeedbackHologram.
        super().__init__(shape, target_ij=None, cameraslm=cameraslm, **kwargs)

        # Parse null_region after __init__
        if basis == "ij" and null_region is not None:
            # Transformation order of zero to prevent nan-blurring in MRAF cases.
            self.null_region_knm = (
                self.ijcam_to_knmslm(null_region, out=self.null_region_knm, order=0)
                != 0
            )

        # If we have an input for null_region_radius_frac, then force the null region to
        # exclude higher order k-vectors according to the desired exclusion fraction.
        if null_region_radius_frac is not None:
            # Build up the null region pattern if we have not already done the transform above.
            if self.null_region_knm is None:
                self.null_region_knm = cp.zeros(self.shape, dtype=bool)

            # Make a circle, outside of which the null_region is active.
            xl = cp.linspace(-1, 1, self.null_region_knm.shape[0])
            yl = cp.linspace(-1, 1, self.null_region_knm.shape[1])
            (xg, yg) = cp.meshgrid(xl, yl)
            mask = cp.square(xg) + cp.square(yg) > null_region_radius_frac**2
            self.null_region_knm[mask] = True

        # Fill the target with data.
        self.set_target(reset_weights=True)

    def __len__(self):
        """
        Overloads len() to return the number of spots in this :class:`SpotHologram`.

        Returns
        -------
        int
            The length of :attr:`spot_amp`.
        """
        return self.spot_knm.shape[1]

    # Target update.
    @staticmethod
    def make_rectangular_array(
        shape,
        array_shape,
        array_pitch,
        array_center=None,
        basis="knm",
        orientation_check=False,
        **kwargs,
    ):
        """
        Helper function to initialize a rectangular 3D array of spots, with certain size and pitch.

        Note
        ~~~~
        The array can be in SLM k-space coordinates or in camera pixel coordinates, depending upon
        the choice of ``basis``. For the ``"ij"`` basis, ``cameraslm`` must be included as one
        of the ``kwargs``. See :meth:`__init__()` for more ``basis`` information.

        Important
        ~~~~~~~~~
        Spot positions will be rounded to the grid of computational k-space ``"knm"``,
        to create the target image (of finite size) that algorithms optimize towards.
        Choose ``array_pitch`` and ``array_center`` carefully to avoid undesired pitch
        non-uniformity caused by this rounding.

        Parameters
        ----------
        shape : (int, int)
            Computational shape of the SLM in :mod:`numpy` `(h, w)` form. See :meth:`.SpotHologram.__init__()`.
        array_shape : (int, int, int) OR int
            The size of the rectangular array in number of spots ``(NX, NY, NZ)``.
            If a scalar N is given, assume ``(N, N, N)``.
        array_pitch : (float, float) OR float
            The spacing between spots in the x, y, and z directions ``(pitchx, pitchy, pitchz)``.
            If a single pitch is given, assume ``(pitch, pitch, pitch)``.
        array_center : (float, float) OR None
            The shift of the center of the spot array from the zeroth order.
            Uses ``(x, y, z)`` form in the chosen basis.
            If ``None``, defaults to the position of the zeroth order, converted into the
            relevant basis:

            - If ``"knm"``, this is ``(shape[2], shape[1], shape[0])/2``.
            - If ``"kxy"``, this is ``(0,0,0)``.
            - If ``"ij"``, this is the pixel position of the zeroth order on the
              camera (calculated via Fourier calibration).

        basis : str
            See :meth:`__init__()`.
        orientation_check : bool
            Whether to delete the last two points to check for parity.
        **kwargs
            Any other arguments are passed to :meth:`__init__()`.
        """
        # Parse size and pitch.
        if isinstance(array_shape, REAL_TYPES):
            array_shape = (int(array_shape), int(array_shape), int(array_shape))
        if isinstance(array_pitch, REAL_TYPES):
            array_pitch = (array_pitch, array_pitch, array_pitch)

        # Determine array_center default.
        if array_center is None:
            if basis == "knm":
                array_center = (shape[2] / 2.0, shape[1] / 2.0, shape[0] / 2.0)
            elif basis == "kxy":
                array_center = (0, 0, 0)
            elif basis == "ij":
                assert "cameraslm" in kwargs, "We need an cameraslm to interpret ij."
                cameraslm = kwargs["cameraslm"]
                assert cameraslm is not None, "We need an cameraslm to interpret ij."
                assert "fourier" in cameraslm.calibrations, (
                    "We need an cameraslm with "
                    "fourier-calibrated kxyslm_to_ijcam and ijcam_to_kxyslm transforms "
                    "to interpret ij."
                )

                array_center = toolbox.convert_vector(
                    (0, 0, 0), from_units="kxy", to_units="ij", hardware=cameraslm
                )

        # Make the grid edges.
        x_edge = np.arange(array_shape[0]) - (array_shape[0] - 1) / 2.0
        x_edge = x_edge * array_pitch[0] + array_center[0]
        y_edge = np.arange(array_shape[1]) - (array_shape[1] - 1) / 2.0
        y_edge = y_edge * array_pitch[1] + array_center[1]
        z_edge = np.arange(array_shape[2]) - (array_shape[2] - 1) / 2.0
        z_edge = z_edge * array_pitch[2] + array_center[2]

        # Make the grid lists.
        x_grid, y_grid, z_grid = np.meshgrid(
            x_edge, y_edge, z_edge, sparse=False, indexing="xy"
        )
        x_list, y_list, z_list = x_grid.ravel(), y_grid.ravel(), z_grid.ravel()

        # Delete the last two points if desired and valid.
        if orientation_check and len(x_list) > 2:
            x_list = x_list[:-2]
            y_list = y_list[:-2]
            z_list = z_list[:-2]

        vectors = np.vstack((x_list, y_list, z_list))

        # Return a new SpotHologram.
        return SpotHologram3D(shape, vectors, basis=basis, spot_amp=None, **kwargs)

    def _set_target_spots(self, reset_weights=False):
        """
        Wrapped by :meth:`SpotHologram.set_target()`.
        """
        # Round the spot points to the nearest integer coordinates in knm space.
        self.spot_knm_rounded = np.rint(self.spot_knm).astype(int)

        # Convert these to the other coordinate systems if possible.
        if self.cameraslm is not None:
            self.spot_kxy_rounded = toolbox.convert_vector(
                self.spot_knm_rounded,
                from_units="knm",
                to_units="kxy",
                hardware=self.cameraslm.slm,
                shape=self.shape,
            )

            if "fourier" in self.cameraslm.calibrations:
                self.spot_ij_rounded = self.cameraslm.kxyslm_to_ijcam(
                    self.spot_kxy_rounded
                )
            else:
                self.spot_ij_rounded = None
        else:
            self.spot_kxy_rounded = None
            self.spot_ij_rounded = None

        # Erase previous target in-place.
        if self.null_knm is None:
            self.target.fill(0)
        else:
            # By default, everywhere is "amplitude free", denoted by nan.
            self.target.fill(np.nan)

            # Now we start setting areas where null is desired. First, zero the blanket region.
            if self.null_region_knm is not None:
                self.target[self.null_region_knm] = 0

            # Second, zero the regions around the "null points".
            if self.null_knm is not None:
                all_spots = np.hstack((self.null_knm, self.spot_knm))
                w = int(2 * self.null_radius_knm + 1)

                for ii in range(all_spots.shape[1]):
                    toolbox.imprint(
                        self.target,
                        (np.rint(all_spots[0, ii]), w, np.rint(all_spots[1, ii]), w),
                        0,
                        centered=True,
                        circular=True,
                    )

        # Set all the target pixels to the appropriate amplitude.
        self.target[
            self.spot_knm_rounded[2, :],
            self.spot_knm_rounded[1, :],
            self.spot_knm_rounded[0, :],
        ] = self.spot_amp

        self.target /= Hologram3D._norm(self.target)

        if reset_weights:
            self.reset_weights()

    def set_target(self, reset_weights=False, plot=False):
        """
        From the spot locations stored in :attr:`spot_knm`, update the target pattern.

        Note
        ~~~~
        If there's a ``cameraslm``, updates the :attr:`spot_ij_rounded` attribute
        corresponding to where pixels in the :math:`k`-space where actually placed (due to rounding
        to integers, stored in :attr:`spot_knm_rounded`), rather the
        idealized floats :attr:`spot_knm`.

        Note
        ~~~~
        The :attr:`target` and :attr:`weights` matrices are modified in-place for speed,
        unlike :class:`.Hologram` or :class:`.FeedbackHologram` which make new matrices.
        This is because spot positions are expected to be corrected using :meth:`refine_offsets()`.

        Parameters
        ----------
        reset_weights : bool
            Whether to reset the :attr:`weights` to this new :attr:`target`.
        """
        self._set_target_spots(reset_weights=reset_weights)

    # Weighting and stats.
    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude.
        """
        feedback = self.flags["feedback"]

        # Weighting strategy depends on the chosen feedback method.
        if feedback == "computational":
            # Pixel-by-pixel weighting
            self._update_weights_generic(
                self.weights, self.amp_ff, self.target, nan_checks=True
            )
        else:
            # Integrate a window around each spot, with feedback from respective sources.
            if feedback == "computational_spot":
                amp_feedback = cp.sqrt(
                    analysis.take(
                        cp.square(self.amp_ff),
                        self.spot_knm_rounded,
                        self.spot_integration_width_knm,
                        centered=True,
                        integrate=True,
                        xp=cp,
                    )
                )
            elif feedback == "experimental_spot":
                self.measure(basis="ij")

                amp_feedback = np.sqrt(
                    analysis.take(
                        np.square(
                            np.array(
                                self.img_ij,
                                copy=(False if np.__version__[0] == "1" else None),
                                dtype=self.dtype,
                            )
                        ),
                        self.spot_ij,
                        self.spot_integration_width_ij,
                        centered=True,
                        integrate=True,
                    )
                )
            elif feedback == "external_spot":
                amp_feedback = self.external_spot_amp
            else:
                raise ValueError("Feedback '{}' not recognized.".format(feedback))

            # Update the weights of single pixels.
            self.weights[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]] = (
                self._update_weights_generic(
                    self.weights[
                        self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]
                    ],
                    cp.array(
                        amp_feedback,
                        copy=(False if np.__version__[0] == "1" else None),
                        dtype=self.dtype,
                    ),
                    self.spot_amp,
                    nan_checks=True,
                )
            )

    def _calculate_stats_computational_spot(self, stats, stat_groups=[]):
        """
        Wrapped by :meth:`SpotHologram._update_stats()`.
        """

        if "computational_spot" in stat_groups:
            if self.shape == self.slm_shape:
                # Spot size is one pixel wide: no integration required.
                stats["computational_spot"] = self._calculate_stats(
                    self.amp_ff[
                        self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]
                    ],
                    self.spot_amp,
                    efficiency_compensation=False,
                    total=cp.sum(cp.square(self.amp_ff)),
                    raw="raw_stats" in self.flags and self.flags["raw_stats"],
                )
            else:
                # Spot size is wider than a pixel: integrate a window around each spot
                if cp != np:
                    pwr_ff = cp.square(self.amp_ff)
                    pwr_feedback = analysis.take(
                        pwr_ff,
                        self.spot_knm,
                        self.spot_integration_width_knm,
                        centered=True,
                        integrate=True,
                        xp=cp,
                    )

                    stats["computational_spot"] = self._calculate_stats(
                        cp.sqrt(pwr_feedback),
                        self.spot_amp,
                        xp=cp,
                        efficiency_compensation=False,
                        total=cp.sum(pwr_ff),
                        raw="raw_stats" in self.flags and self.flags["raw_stats"],
                    )
                else:
                    pwr_ff = np.square(self.amp_ff)
                    pwr_feedback = analysis.take(
                        pwr_ff,
                        self.spot_knm,
                        self.spot_integration_width_knm,
                        centered=True,
                        integrate=True,
                    )

                    stats["computational_spot"] = self._calculate_stats(
                        np.sqrt(pwr_feedback),
                        self.spot_amp,
                        xp=np,
                        efficiency_compensation=False,
                        total=np.sum(pwr_ff),
                        raw="raw_stats" in self.flags and self.flags["raw_stats"],
                    )

    def _calculate_stats_experimental_spot(self, stats, stat_groups=[]):
        """
        Wrapped by :meth:`SpotHologram._update_stats()`.
        """

        if "experimental_spot" in stat_groups:
            self.measure(basis="ij")

            pwr_img = np.square(self.img_ij)

            pwr_feedback = analysis.take(
                pwr_img,
                self.spot_ij,
                self.spot_integration_width_ij,
                centered=True,
                integrate=True,
            )

            stats["experimental_spot"] = self._calculate_stats(
                np.sqrt(pwr_feedback),
                self.spot_amp,
                xp=np,
                efficiency_compensation=False,
                total=np.sum(pwr_img),
                raw="raw_stats" in self.flags and self.flags["raw_stats"],
            )

        if "external_spot" in stat_groups:
            pwr_feedback = np.square(
                np.array(
                    self.external_spot_amp,
                    copy=(False if np.__version__[0] == "1" else None),
                    dtype=self.dtype,
                )
            )
            stats["external_spot"] = self._calculate_stats(
                np.sqrt(pwr_feedback),
                self.spot_amp,
                xp=np,
                efficiency_compensation=False,
                total=np.sum(pwr_feedback),
                raw="raw_stats" in self.flags and self.flags["raw_stats"],
            )

    def _update_stats(self, stat_groups=[]):
        """
        Calculate statistics corresponding to the desired ``stat_groups``.

        Parameters
        ----------
        stat_groups : list of str
            Which groups or types of statistics to analyze.
        """
        stats = {}

        self._calculate_stats_computational(stats, stat_groups)
        self._calculate_stats_experimental(stats, stat_groups)
        self._calculate_stats_computational_spot(stats, stat_groups)
        self._calculate_stats_experimental_spot(stats, stat_groups)

        self._update_stats_dictionary(stats)

    def refine_offset(self, img=None, basis="kxy", force_affine=True, plot=False):
        """
        Hones the positions of the produced spots toward the desired targets to compensate for
        Fourier calibration imperfections. Works either by moving camera integration
        regions to the positions where the spots ended up (``basis="ij"``) or by moving
        the :math:`k`-space targets to target the desired camera pixels
        (``basis="knm"``/``basis="kxy"``). This should be run at the user's request
        inbetween :meth:`optimize` iterations.

        Parameters
        ----------
        img : numpy.ndarray OR None
            Image measured by the camera. If ``None``, defaults to :attr:`img_ij` via :meth:`measure()`.
        basis : str
            The correction can be in any of the following bases:

            - ``"ij"`` changes the pixel that the spot is expected at,
            - ``"kxy"``, ``"knm"`` changes the k-vector which the SLM targets.

            Defaults to ``"kxy"``. If basis is set to ``None``, no correction is applied
            to the data in the :class:`SpotHologram`.
        force_affine : bool
            Whether to force the offset refinement to behave as an affine transformation
            between the original and refined coordinate system. This helps to tame
            outliers. Defaults to ``True``.
        plot : bool
            Enables debug plots.

        Returns
        -------
        numpy.ndarray
            Spot shift in the ``"ij"`` basis for each spot.
        """
        # If no image was provided, get one from cache.
        if img is None:
            self.measure(basis="ij")
            img = self.img_ij

        # Take regions around each point from the given image.
        regions = analysis.take(
            img,
            self.spot_ij,
            self.spot_integration_width_ij,
            centered=True,
            integrate=False,
        )

        # Fast version; have to iterate for accuracy.
        shift_vectors = analysis.image_positions(regions)
        shift_vectors = np.clip(
            shift_vectors,
            -self.spot_integration_width_ij / 4,
            self.spot_integration_width_ij / 4,
        )

        # Store the shift vector before we force_affine.
        sv1 = self.spot_ij + shift_vectors

        if force_affine:
            affine = analysis.fit_affine(
                self.spot_ij, self.spot_ij + shift_vectors, plot=plot
            )
            shift_vectors = (
                np.matmul(affine["M"], self.spot_ij) + affine["b"]
            ) - self.spot_ij

        # Record the shift vector after we force_affine.
        sv2 = self.spot_ij + shift_vectors

        # Plot the above if desired.
        if plot:
            mask = analysis.take(
                img,
                self.spot_ij,
                self.spot_integration_width_ij,
                centered=True,
                integrate=False,
                return_mask=True,
            )

            plt.figure(figsize=(12, 12))
            plt.imshow(img * mask)
            plt.scatter(sv1[0, :], sv1[1, :], s=200, fc="none", ec="r")
            plt.scatter(sv2[0, :], sv2[1, :], s=300, fc="none", ec="b")
            plt.show()

        # Handle the feedback applied from this refinement.
        if basis is not None:
            if basis == "kxy" or basis == "knm":
                # Modify k-space targets. Don't modify any camera spots.
                self.spot_kxy = self.spot_kxy - (
                    self.cameraslm.ijcam_to_kxyslm(shift_vectors)
                    - self.cameraslm.ijcam_to_kxyslm((0, 0))
                )
                self.spot_knm = toolbox.convert_vector(
                    self.spot_kxy,
                    to_units="kxy",
                    from_units="knm",
                    hardware=self.cameraslm.slm,
                    shape=self.shape,
                )
                self.set_target(reset_weights=True)
            elif basis == "ij":
                # Modify camera targets. Don't modify any k-vectors.
                self.spot_ij = self.spot_ij + shift_vectors
            else:
                raise Exception("Unrecognized basis '{}'.".format(basis))

        return shift_vectors
