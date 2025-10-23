# Workflow

The overall workflow of processing an image with AFMSlicer is shown
below.[^mermaid]

```mermaid
---
title: AFMSlicer
config:
  theme: 'dark'
---
flowchart RL

LoadScan("`**topostats.io.LoadScan()**
           Load the image from AFM file`")
Filter("`**topostats.filter.Filters()**
         Filter and flatten the image`")
AFMSlicer("`**afmslicer.slice.Slice()**
            Convert 2D numpy array to 3D numpy array of slices`")
MinMaxHeights("`Determine Min/Max Heights`")
SliceRanges("`Ranges for Slices given user Requested number`")
BinaryMask("`Binary Mask for values within range`")

subgraph Slicing
   MinMaxHeights --> SliceRanges
   SliceRanges --> BinaryMask
end

subgraph Filtering
end


subgraph master
    LoadScan --> Filter
    Filter --> AFMSlicer
end


%%style LoadScan fill:#f2e88c,stroke:#000000
%%style Filter fill:#9ddbd7,stroke:#000000
%%style AFMSlicer fill:#f4a666,stroke:#000000

```

[^mermaid]:
    If modifying this diagram it is recommended to copy and paste the existing
    code and adjust in the [mermaid.live][mermaid_live] online tool.

[mermaid_live]: https://mermaid.live
