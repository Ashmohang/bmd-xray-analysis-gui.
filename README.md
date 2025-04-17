# BMD X-ray Analysis: GUI Tool

An interactive Python-based GUI application for automated Bone Mineral Density (BMD) estimation from forearm X-rays. This tool includes phantom removal, cropping, CLAHE-based segmentation, region-of-interest detection, and soft tissue–bone intensity estimation — all wrapped in a user-friendly interface using Tkinter.

## 🖥️ Features

- Load anonymized X-ray images
- Step-by-step visualization:
  1. Phantom removal
  2. Image cropping
  3. CLAHE and segmentation
  4. Sobel edge detection
  5. Region of interest analysis with bounding boxes
- Automated ROI detection and visualization

## 📷 Sample

<img src="sample_output.png" width="700">

## 📦 Dependencies

Install using pip:

```bash
pip install -r requirements.txt
```

**Requirements:**

- numpy
- matplotlib
- opencv-python
- scikit-image
- scipy
- tkinter (comes built-in with most Python installations)

## 🚀 Run the GUI

```bash
python gui_app/bmd_gui_app.py
```

## 📁 File Structure

- `gui_app/bmd_gui_app.py`: Main application
- `sample_data/`: (Optional) Add your test images here
- `requirements.txt`: Dependencies

## 🧑‍💻 Author

Ashwin  
MSc Diagnostics, Data & Digital Health  
University of Warwick
