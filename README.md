# BMD X-ray Analysis: GUI Tool

This repository contains a Python-based GUI application for automated Bone Mineral Density (BMD) estimation from forearm X-ray images. The tool enables end-to-end analysis, including phantom removal, image preprocessing, segmentation, and region-based intensity measurement for aBMD calculation — all integrated into an interactive interface using Tkinter.

## Features

- Load and analyze anonymized forearm X-ray images
- Step-by-step processing pipeline:
  - Phantom removal
  - Image cropping
  - Contrast enhancement and segmentation using CLAHE and multi-Otsu thresholding
  - Edge detection using Sobel filters
  - Region-of-interest (ROI) detection and intensity analysis
- Interactive visualization with bounding boxes for bone and soft tissue regions
- Designed for reproducible and semi-automated analysis in clinical imaging research

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

Make sure your Python installation includes Tkinter (comes pre-installed with most standard distributions).

## Usage

To launch the application:

```bash
python gui_app/bmd_gui_app.py
```

You can then load an image and proceed through each step using the interface.

## Repository Structure

```
bmd-xray-analysis-gui/
├── gui_app/
│   └── bmd_gui_app.py          # Main application script
├── requirements.txt            # Project dependencies
├── README.md                   # Project description and usage
```

## Author

**Ashwin Nambiar**  
MSc Diagnostics, Data and Digital Health  
University of Warwick
