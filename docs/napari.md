# Napari

[Napari][napari] is an extensible framework for viewing, processing and manipulating images and we have developed a
plugin to use AFMslicer in Napari, [napari-afmslicer][napari-afmslicer].

## Installation

We publish [napari-afmslicer][napari-afmslicer] to PyPI and in turn it appears in the [Napari Hub][napari-hub]. You can
then install it from the _Plugins > Install/Uninstall Plugins_ menu item of Napari.

### Dependencies

We include a small selection of plugins are included as dependencies and will be installed automatically.

- [napari-afmreader][napari-afmreader] - used to load a range of Atomic Force Microscopy images.
- [napari-crop][napari-crop] - used for cropping images (typically prior to filtering and slicing).
- [napari-skimage][napari-skimage] - Napari interface to [Scikit Image][skimage] functions.
- [napari-topostats][napari-topostats] - required as some of the functionality is reused.

The `napari-crop` and `napari-skimage` plugins in particular may be of use to further pre-process images prior to slicing.

## Usage

Once the plugin has been installed you will find a menu item _Plugins > AFMSlicer >_ from which you can enable each
component of the plugin. Currently there are three components, each of which can be enabled independently.

- 3D Viewer
- Filter
- Slicing

Load each of the widgets by navigating to _Plugins > AFMSlicer_ and selecting each of _AFMSlicer Filter_, _AFMSlicer 3D
Viewer_ and _AFMSlicer Slicing_ which will add the widgets to the right-hand side of Napari. A typical workflow will
involve...

1. Filtering an image and removing scars
2. Running slicing.

Hovering over any of the text for options will show a short explanation of what it does.

### Filtering

The parameters for filtering can be configured easily. Unchecking _Remove scars_ will prevent scar removal from running
completely. Once you have adjusted the parameters to your preference click on the _Filter_ button and the image will be
filtered and a new layer will appear listed on the left and automatically selected showing the effects of filtering. It
will have the original images name suffixed with `_filtered`

### Slicing

Under the _Slicing_ widget select the filtered image from the pull-down list to the right of _Image_", select the number
of _Slices_ to create, the _Segmentation method_ and other options, in particular the _Output directory_ should be
explicitly set. When ready click the _Slice Image and Summarise_, the image will be sliced and summarised with the
output saved to the parameter in the _Output directory_ field.

## Extending

If you are contributing to and extending AFMSlicer you will likely want to include the changes and new features in the
Napari. There is extensive documentation on developing [Napari plugins][napari-plugins] which leverage the
[magicgui][magicgui] package to facilitate creation of "widgets".

A worked example is included for illustrative purposes and should help with adding or extending options. You can always
refer to the [source code][napari-afmslicer] for more details when extending.

### 3D Viewer Widget

Napari leverages [magicgui][magicgui] to make it easier to construct "widgets" and there are a number of
[examples][magicgui-examples] in the documentation. Here is a simplified version that I actually understand.

We leverage the `@magicfactory()` decorator and for each argument of the function that is being decorated (bar the
`image` argument which Napari seems to sort out on its own) we define a dictionary with a bare minimum of a `label`
key. The value of this `label` key should be a string and is the text that is displayed against the option. If there are
not other key/value pairs then this includes a simple text box which is appropriate for a boolean parameter. If a
numerical value is required to be input then you can include key the keys `min`, `max` and `step` which define the range
and the increments the slider adjusts in. An illustrative example is shown below...

``` python
"""View images in three dimensions."""

from typing import TYPE_CHECKING

import napari.types
from magicgui import magic_factory
from napari import current_viewer  # pylint: disable=no-name-in-module
from napari.layers import Image
from napari_topostats.utils import afm2stack

if TYPE_CHECKING:
    import napari


@magic_factory(
    by_slices={"label": "Whether to stack by slices."},
    numslices={"label": "Number of slices.", "min": 20, "max": 2000, "step": 1},
    resolution={"label": "Resolution.", "min": 1.0, "max": 100.0, "step": 0.1},
)
def view_3d(
    image: Image,
    by_slices: bool = True,
    numslices: int = 255,
    resolution: float = 1.0,
) -> napari.types.LayerDataTuple:
    """
    View image in three dimensions.

    Parameters
    ----------
    image : Image
        Image to be viewed in three dimensions.
    by_slices : bool
        Whether to stack by slices (default ``True``). If ``False`` then ``resolution`` is used.
    numslices : int
        Number of slices to create.
    resolution : float
        The resolution/distance between each slice, by default 1.0.

    Returns
    -------
    napari.types.LayerDataTuple:
        Modified image in three-dimensions.
    """
    three_dimensions = afm2stack(
        image=image.data,
        by_slices=by_slices,
        numslices=numslices,
        resolution=resolution,
    )
    viewer = current_viewer()
    viewer.dims.ndisplay = 3
    return (
        three_dimensions,
        {"name": f"{image.name}_3D"},
        "image",
    )
```

Here the plugin component for viewing the image in 3D has three "widgets", a tick box for whether to stack slices which
is stored in `by_slices`, the number of slices (`numslices`) which can be between `20` and `2000` in steps of `1` and
the resolution which can be between `1.0` and `100.0` in steps of `0.1`. These map directly to the arguments of the
`view_3d()` function which is a thin wrapper to `afm2stack()`. In this instance the current viewer is obtained and the
number of display dimensions is increased to `3` (as this renders a 2D image in 3D). Finally a `LayerDataTuple` is returned.

To add this to the menus in the Napari GUI you add the following to `napari.yaml`.

``` yaml
contributions:
  commands:
    - id: napari-afmslicer.view_3d_widget
      python_name: napari_afmslicer.view_3d_widget:view_3d
      title: AFMSlicer 3D Viewer
  widgets:
    - command: napari-afmslicer.view_3d_widget
      display_name: AFMSlicer 3D Viewer
  menus:
    napari/layers/filters:
      - submenu: filtering_submenu
    filtering_submenu:
      - command: napari-afmslicer.view_3d_widget

  submenus:
    - id: filtering_submenu
      label: Filtering
```

[magicgui]: https://pyapp-kit.github.io/magicgui/
[magicgui-examples]: https://pyapp-kit.github.io/magicgui/generated_examples/napari/napari_parameter_sweep/
[napari]: https://napari.org/stable/
[napari-crop]: https://napari-hub.org/plugins/napari-crop.html
[napari-skimage]: https://napari-hub.org/plugins/napari-skimage.html
[napari-hub]: https://napari-hub.org/
[napari-plugins]: https://napari.org/stable/plugins/index.html
[napari-afmreader]: https://github.com/afm-spm/napari-afmreader/
[napari-afmslicer]: https://github.com/afm-spm/napari-afmslicer/
[napari-crop]: https://github.com/afm-spm/napari-crop/
[napari-skimage]: https://github.com/afm-spm/napari-skimage/
[napari-topostats]: https://github.com/afm-spm/napari-topostats/
[skimage]: https://scikit-image.org/
