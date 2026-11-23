# Workflow

The overall workflow of processing an image with AFMSlicer is shown
below.[^mermaid]

```mermaid
---
title: AFMSlicer Overview
config:
  theme: 'dark'
---
flowchart LR

LoadScan("`**topostats.io.LoadScan()**
           Load the image from AFM file`")
Filter("`**topostats.filter.Filters()**
           Filter and flatten the image`")
AFMSlicer("`**afmslicer.AFMSlicer()**
           Create AFMSlicer object, slice, segment, region properties.`")
Stats("`**afmslicer.statistics**
           Calculate statistics`")

LoadScan --> Filter
Filter --> AFMSlicer
AFMSlicer --> Stats

%%style LoadScan fill:#f2e88c,stroke:#000000
%%style Filtering fill:#9ddbd7,stroke:#000000
%%style AFMSlicer fill:#f4a666,stroke:#000000
%%style Stats fill:#f4a666,stroke:#000000

```

```mermaid
---
title: AFMSlicer Detailed
config:
  theme: 'dark'
---
flowchart LR

Loading("`**TopoStats.LoadScan** (**AFMReader**)
          Load scans for processing (read metadata)`")
Filtering("`**TopoStats.Filter()**
          Flatten, remove scars and Gaussian blurring`")
MinMaxHeights("`Determine Min/Max Heights`")
SliceRanges("`**AFMSlicer.slices**
          Ranges for Slices given user Requested number`")
Slicing("`**slicer.slicer()**
          Create slices.`")
BinaryMask("`**slicer.mask_slices()**
          Binary Mask for values within range`")
SegmentSlices("`**slicer.segment_slices()**
          Segment individual slices to detect unique objects`")
RegionProperties("`**slicer.region_properties_by_slice()**
          Calculate statistics on each object across all slices`")
Clean("`**slicer.mask_small_artefacts_all_layers()**
          Remove all objects < minimum size`")
Count("`**statistics.count_pores()**
          Count pores (objects) per layer.`")
CalculateArea("`**statistics.area_pores()**
          Calculate area of remaining objects`")
CumulativeArea("`**statistics.cumulative_areas()**
          Cumulative area and sorting`")
HCFA("`**statistics.hcfa()**
          Calculate Half-Cumulative Fraction of Area (HCFA)`")
PDF("`**statistics.calculate_pdf()**
          Fit Gaussian Probability Density Function (PDF)`")
FWHM("`**statistics.full_width_half_max()**
          Full-Width Half Max of areas`")
Plot("`Generate plots summarising objects per layer and areas`")

LoadScan --> Filter
Filter --> slicer
slicer --> statistics

subgraph LoadScan
   Loading
end
subgraph Filter
    Filtering
end

subgraph slicer
   MinMaxHeights --> SliceRanges
   SliceRanges --> Slicing
   Slicing --> BinaryMask
   BinaryMask --> SegmentSlices
   SegmentSlices --> RegionProperties
   RegionProperties --> Clean
   Clean --> Count
end

subgraph statistics
   CalculateArea --> CumulativeArea
   CumulativeArea --> HCFA
   HCFA --> PDF
   PDF --> FWHM
   FWHM --> Plot
end

```

[^mermaid]:
    If modifying this diagram it is recommended to copy and paste the existing
    code and adjust it in the [mermaid.live][mermaid_live] online tool.

[mermaid_live]: https://mermaid.live
