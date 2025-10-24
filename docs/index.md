# AFMSlicer

[AFMSlicer][afmslicer] is a program for processing and analysing [Atomic Force Microscopy][afm] images of bacterial cell
walls.

It leverages the [TopoStats][topostats] package for loading and flattening/filtering images and as such follows a
similar structure in terms of configuration and design patterns.

After filtering/flattening of an image it is then "sliced" into an 3-D [numpy][numpy] array of masks which are analysed
to obtain statistics on the structure of images at different heights.

[afm]: https://en.wikipedia.org/wiki/Atomic_force_microscopy
[afmslicer]: https://afmslicer.readthedocs.org/
[numpy]: https://numpy.org
[topostats]: https://AFM-SPM.github.io/TopoStats
