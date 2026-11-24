# AFMSlicer

[AFMSlicer][afmslicer] is a program for processing and analysing [Atomic Force Microscopy][afm] images of bacterial cell
walls.

It leverages the [TopoStats][topostats] package for loading and flattening/filtering images and as such follows a
similar structure in terms of configuration and design patterns.

After filtering/flattening an image it is then "sliced" into an 3-D [numpy][numpy] array of masks and objects are
identified within each layer. These are analysed their structure and aggregate statistics are calculated across layers.

[afm]: https://en.wikipedia.org/wiki/Atomic_force_microscopy
[afmslicer]: https://afmslicer.readthedocs.org/
[numpy]: https://numpy.org
[topostats]: https://AFM-SPM.github.io/TopoStats
