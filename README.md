<p align="center">
  <img src="assets/logo.png" alt="Region Search logo: a stack of BigBrain coronal sections with a magnifier over the hippocampus" width="180">
</p>

<h1 align="center">Hide and Seek: Brain Edition</h1>

<p align="center">
  Search any brain region across hundreds of individual slices.
</p>

---

## The problem

Researchers who study a specific brain structure, such as the hippocampus, a sulcus or a brainstem nucleus, often
need to find it across hundreds of brain sections. Today this means scrolling through the stack and inspecting each
section by hand.

This project builds a search system for that task. The user gives either:

- **an ROI**: a box or outline drawn on one section, or
- **a region name**: for example "hippocampus" or "inferior olive".

The system returns the sections that most likely contain the region, ranked, with the matching location on each one.

## Goals

1. Build a first search method with image embeddings, ROI matching, or other image-analysis techniques.
2. Given a brain image or a stack of sections and an ROI, identify the sections most likely to contain the region.
3. Explore search by region name as well as by user-selected ROI.

## Requirements

- Basic Python and image processing, or willingness to learn.
- Minimal brain anatomy, or willingness to learn.

## Possible approaches

These are starting points, not a fixed plan.

| Approach | Idea | Notes |
|---|---|---|
| Template matching | Slide the ROI over each section and score it with normalised cross-correlation | Simple baseline. Can confuse neighbouring grey-matter structures |
| Hand-crafted features | Texture and shape descriptors (HOG, LBP, Gabor filters, SIFT/ORB keypoints) per patch, compared with a distance | Less sensitive to stain intensity than raw pixels |
| Registration | Align neighbouring sections, then carry the ROI from section to section | Uses the order of the stack. Errors can accumulate along it |
| Image embeddings | Embed patches with a pretrained vision or histopathology model and search by nearest neighbour | Fast for many queries once the embeddings are computed |
| Name search | Map a region name to a location with an atlas registered to the stack, then run ROI search from there | Needs an atlas in the same space as the sections |

**Evaluation idea:** for a few structures, mark the range of sections where each one is visible. Then report
precision@k and average precision of the ranked sections.

## Data

A suggested starting dataset is the [BigBrain](https://bigbrainproject.org) 2015 release (Amunts et al., *Science*
2013): one human brain cut into 7404 coronal sections of 20 µm, stained for cell bodies (Merker stain) and
reconstructed in 3D. Download it from <https://ftp.bigbrainproject.org/bigbrain-ftp/>. Keep data files in `data/`;
they are not committed.

## Repository layout

```
assets/             logo (png, svg)
data/               local data (not committed)
notebooks/          exploration notebooks
results/            search outputs (not committed)
scripts/            command-line scripts
src/region_search/  search code
requirements.txt
```

## Credits and licence

The logo is drawn from the BigBrain tissue classification. BigBrain: Amunts K, Lepage C, Borgeat L, et al.
BigBrain: an ultrahigh-resolution 3D human brain model. *Science* 340(6139):1472-1475, 2013, licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), so the logo carries the same licence.
A licence for the code has not been chosen yet.
