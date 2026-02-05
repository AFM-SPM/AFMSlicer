"""Filtering images for AFMSlicer processing."""

from __future__ import annotations

from topostats import scars
from topostats.classes import TopoStats
from topostats.filters import Filters


class SlicingFilter(Filters):
    """
    Filtering and flattening of images for AFMSlicer.

    Parameters
    ----------
    topostats_object : TopoStats
        A ``TopoStats`` object of the image to be filtered and flattened.
    row_alignment_quantile : float, optional
        Quantile on which to align values, default is ``0.5`` (i.e. the median).
    gaussian_size : float, optional
        Amount of Gaussian blurring to perform.
    gaussian_mode : str, optional
        Method of Gaussian blurring to use.
    remove_scars : dict[str, bool | int | float]
        Whether to remove scars or not. This is performed using TopoStats' scar removal method.
    """

    def __init__(
        self,
        topostats_object: TopoStats,
        row_alignment_quantile: float | None = None,
        gaussian_size: float | None = None,
        gaussian_mode: str | None = None,
        remove_scars: dict[str, bool | int | float] | None = None,
    ):
        """
        Initialise the class.

        This class inherits from the ``topostats.Filters()`` class and methods are reused to filter and flatten the
        image.

        Parameters
        ----------
        topostats_object : TopoStats
            A ``TopoStats`` object of the image to be filtered and flattened.
        row_alignment_quantile : float, optional
            Quantile on which to align values, default is ``0.5`` (i.e. the median).
        gaussian_size : float, optional
            Amount of Gaussian blurring to perform.
        gaussian_mode : str, optional
            Method of Gaussian blurring to use.
        remove_scars : dict[str, bool | int | float]
            Whether to remove scars or not. This is performed using TopoStats' scar removal method.

        Examples
        --------

        The final image after all stages is stored in the ``SlicingFilter.image`` dictionary with the key
        ``gaussian_filtered``.

        >>> import numpy as np
        >>> from topostats.classes import TopoStats()
        >>> from afmslicer.filter import SlicingFilter

        >>> rng = np.random.default_rng(seed=32424308)
        >>> small_array = rng.random((5, 5), dtype=np.float64)

        >>> topostats_object = TopoStats(
        >>>     image_original=small_array,
        >>>     filename="small_array",
        >>>     pixel_to_nm_scaling=1.0,
        >>>     img_path="./")

        >>> filter_config = {
        >>>    "row_alignment_quantile": 0.5,
        >>>    "gaussian_size": 1.012139,
        >>>    "gaussian_mode": nearest,
        >>>    "remove_scars": {
        >>>        "run": False,
        >>>        "removal_iterations": 2 # Number of times to run scar removal.
        >>>        "threshold_low": 0.250 # lower values make scar removal more sensitive
        >>>        "threshold_high": 0.666 # lower values make scar removal more sensitive
        >>>        "max_scar_width": 4 # Maximum thickness of scars in pixels.
        >>>        "min_scar_length": 16
        >>>    }
        >>> }

        >>> slicing_filter = SlicingFilter(
        >>>     topostats_object=topostats_object,
        >>>     **filter_config)
        >>> slicing_filter.filter_image()

        >>> slicing_filter.images["gaussian_filtered"]
        """
        super().__init__(topostats_object)
        self.gaussian_size = gaussian_size
        self.gaussian_mode = gaussian_mode
        self.row_alignment_quantile = row_alignment_quantile
        self.remove_scars = remove_scars
        self.images = {
            "pixels": self.topostats_object.image_original,
            "median_flatten": None,
            "tilt_removal": None,
            "quadratic_removal": None,
            "scar_removal": None,
            "zero_average_background": None,
            "gaussian_filtered": None,
        }

    def filter_image(self):
        """
        Filter the image.

        The following steps are performed to filter the image.

        * Median flattening.
        * Tilt removal.
        * Quadratic removal.
        * Nonlinear polynomial removal.
        * Scar removal (optional).
        * Zero average background.
        * Gaussian filtering.

        The methods used are inherited from ``TopoStats.Filters`` class.
        """
        self.images["median_flatten"] = self.median_flatten(
            self.images["pixels"],
            mask=None,
            row_alignment_quantile=self.row_alignment_quantile,
        )
        self.images["tilt_removal"] = self.remove_tilt(
            self.images["median_flatten"], mask=None
        )
        self.images["quadratic_removal"] = self.remove_quadratic(
            self.images["tilt_removal"], mask=None
        )
        self.images["nonlinear_polynomial_removal"] = self.remove_nonlinear_polynomial(
            self.images["quadratic_removal"], mask=None
        )
        if self.remove_scars.pop("run"):
            self.images["scar_removal"], _ = scars.remove_scars(
                self.images["nonlinear_polynomial_removal"],
                filename=self.filename,
                **self.remove_scars,
            )
        else:
            self.images["scar_removal"] = self.images["nonlinear_polynomial_removal"]
        self.images["zero_average_background"] = self.average_background(
            self.images["scar_removal"], mask=None
        )
        self.images["gaussian_filtered"] = self.gaussian_filter(
            self.images["zero_average_background"]
        )
