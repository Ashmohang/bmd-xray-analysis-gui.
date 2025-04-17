import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu, sobel
from skimage.exposure import equalize_adapthist
from scipy.ndimage import label, find_objects
import matplotlib

# Setting the Matplotlib backend to TkAgg for Tkinter compatibility
matplotlib.use('TkAgg')


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bone Density Analysis Tool")
        self.image = None

        # UI Elements
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.step1_button = tk.Button(root, text="Step 1: Phantom Removal", command=self.step1, state=tk.DISABLED)
        self.step1_button.pack()

        self.step2_button = tk.Button(root, text="Step 2: Crop Image", command=self.step2, state=tk.DISABLED)
        self.step2_button.pack()

        self.step3_button = tk.Button(root, text="Step 3: CLAHE & Segmentation", command=self.step3, state=tk.DISABLED)
        self.step3_button.pack()

        self.step4_button = tk.Button(root, text="Step 4: Sobel Edge Detection", command=self.step4, state=tk.DISABLED)
        self.step4_button.pack()

        self.step5_button = tk.Button(root, text="Step 5: Connected Components & ROI", command=self.step5, state=tk.DISABLED)
        self.step5_button.pack()

        self.canvas = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.image is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            self.step1_button.config(state=tk.NORMAL)

    def step1(self):
        if self.image is None:
            messagebox.showerror("Error", "Load an image first")
            return

        self.phantom_removed_image = self.image.copy()

        x_start, x_end = 300, 800
        y_start, y_end = 700, 1800
        phantom = self.phantom_removed_image[y_start:y_end, x_start:x_end]
        for row_idx in range(phantom.shape[0]):
            row = phantom[row_idx, :]
            start_index, end_index = -1, -1
            for i in range(1, len(row)):
                if row[i-1] < 128.55 and row[i] >= 128.55:
                    start_index = i
                elif row[i-1] > 128.55 and row[i] <= 128.55 and start_index != -1:
                    end_index = i
                    break
            if start_index != -1 and end_index != -1:
                phantom[row_idx, start_index:end_index + 1] = 120

        self.phantom_removed_image[y_start:y_end, x_start:x_end] = phantom

        self.show_image(self.phantom_removed_image, "Phantom Removed Image")
        self.step2_button.config(state=tk.NORMAL)

    def step2(self):
        if self.phantom_removed_image is None:
            messagebox.showerror("Error", "Run Step 1 first")
            return

        roi_start_row, roi_end_row = 0, 1500
        roi_start_col, roi_end_col = 650, 1500
        roi = self.phantom_removed_image[roi_start_row:roi_end_row, roi_start_col:roi_end_col]

        row_sum_roi = np.sum(roi, axis=1)

        roi_start_row_index = 50
        roi_end_row_index = 2450

        row_sum_roi_sliced = row_sum_roi[roi_start_row_index:roi_end_row_index + 1]
        max_row_index_in_sliced = np.argmax(row_sum_roi_sliced)
        min_row_index_from_max_in_sliced = np.argmin(row_sum_roi_sliced[max_row_index_in_sliced:])
        max_row_index = roi_start_row_index + max_row_index_in_sliced
        min_row_index = max_row_index + min_row_index_from_max_in_sliced

        crop_start_row_index = min_row_index - 50
        if crop_start_row_index < 0:
            crop_start_row_index = 0

        self.cropped_image = self.phantom_removed_image[crop_start_row_index:, :]

        self.show_image(self.cropped_image, "Cropped Image")
        self.step3_button.config(state=tk.NORMAL)

    def step3(self):
        if self.cropped_image is None:
            messagebox.showerror("Error", "Run Step 2 first")
            return

        clahe = equalize_adapthist(self.cropped_image, clip_limit=0.09)
        thresholds = threshold_multiotsu(clahe, classes=4)
        self.segmentedImage = np.digitize(clahe, bins=thresholds)

        mask = np.where(self.segmentedImage == 3, 1, 0)
        self.binary_mask = mask * 255

        self.show_image(self.binary_mask, "Binary Mask (Bone Region)")
        self.step4_button.config(state=tk.NORMAL)

    def step4(self):
        if self.binary_mask is None:
            messagebox.showerror("Error", "Run Step 3 first")
            return

        col_index, drop_row_index = self.find_pattern_in_image(self.binary_mask)
        if col_index != -1 and drop_row_index != -1:
            pixels_per_mm = 8.3333
            roi_length_mm = 50
            roi_length_pixels = int(roi_length_mm * pixels_per_mm)

            row_index_50mm = drop_row_index + roi_length_pixels
            roi_5mm_length_mm = 5
            roi_5mm_length_pixels = int(roi_5mm_length_mm * pixels_per_mm)
            row_index_5mm_from_50mm = row_index_50mm + roi_5mm_length_pixels

            roi = self.binary_mask[row_index_50mm:row_index_5mm_from_50mm, :]
            sobel_x = sobel(roi, axis=1)
            sobel_y = sobel(roi, axis=0)
            edges = np.hypot(sobel_x, sobel_y)
            edges = (edges > 0).astype(np.uint8) * 255

            self.rows, self.cols = np.where(edges == 255)
            if len(self.rows) > 0 and len(self.cols) > 0:
                self.top_left = (np.min(self.cols), np.min(self.rows) + row_index_50mm)
                self.bottom_right = (np.max(self.cols), np.max(self.rows) + row_index_50mm)

        self.show_image(self.binary_mask, "Edges Detected with Sobel")
        self.step5_button.config(state=tk.NORMAL)

    def step5(self):
        if self.binary_mask is None:
            messagebox.showerror("Error", "Run Step 4 first")
            return

        objects = self.extract_bright_regions(self.binary_mask, self.top_left[1], self.bottom_right[1])

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # Create 3 subplots for input image, mask, and final output

        # Displaying the original input image
        axes[0].imshow(self.image, cmap='gray')
        axes[0].set_title('Original Input Image')
        axes[0].set_xlabel('X-axis')
        axes[0].set_ylabel('Y-axis')

        # Displaying the binary mask image with bounding boxes (for reference)
        axes[1].imshow(self.binary_mask, cmap='gray')
        axes[1].set_title('Binary Mask (Bone Region)')
        axes[1].set_xlabel('X-axis')
        axes[1].set_ylabel('Y-axis')

        for obj in objects:
            top_left = (obj[1].start, obj[0].start + self.top_left[1])
            bottom_right = (obj[1].stop, obj[0].stop + self.top_left[1])
            rect = plt.Rectangle(top_left, bottom_right[0] - top_left[0], bottom_right[1] - top_left[1],
                                 edgecolor='lime', facecolor='none', linewidth=2, linestyle='--')
            axes[1].add_patch(rect)

        # Adding the red box (soft tissue) between the two green boxes
        if len(objects) >= 4:
            first_obj = objects[1]
            second_obj = objects[2]

            first_midpoint = ((first_obj[1].start + first_obj[1].stop) // 2, (first_obj[0].start + first_obj[0].stop) // 2)
            second_midpoint = ((second_obj[1].start + second_obj[1].stop) // 2, (second_obj[0].start + second_obj[0].stop) // 2)

            center_midpoint = ((first_midpoint[0] + second_midpoint[0]) // 2, (first_midpoint[1] + second_midpoint[1]) // 2)

            box_size_mm = 5
            box_size_pixels = int(box_size_mm * 8.33)

            top_left_corner = (center_midpoint[0] - box_size_pixels // 2, center_midpoint[1] - box_size_pixels // 2)

            aligned_top_left_corner = (top_left_corner[0], first_obj[0].start + self.top_left[1])
            bottom_right_corner = (aligned_top_left_corner[0] + box_size_pixels, aligned_top_left_corner[1] + box_size_pixels)

            rect = plt.Rectangle(aligned_top_left_corner, box_size_pixels, box_size_pixels,
                                 edgecolor='red', facecolor='none', linewidth=2, linestyle='--')
            axes[1].add_patch(rect)

            # Displaying the cropped image with ROIs
            axes[2].imshow(self.cropped_image, cmap='gray')
            axes[2].set_title('Cropped Image with ROIs')
            axes[2].set_xlabel('X-axis')
            axes[2].set_ylabel('Y-axis')

            for obj in objects:
                top_left = (obj[1].start, obj[0].start + self.top_left[1])
                bottom_right = (obj[1].stop, obj[0].stop + self.top_left[1])
                rect = plt.Rectangle(top_left, bottom_right[0] - top_left[0], bottom_right[1] - top_left[1],
                                     edgecolor='lime', facecolor='none', linewidth=2, linestyle='--')
                axes[2].add_patch(rect)

            new_rect = plt.Rectangle(aligned_top_left_corner, box_size_pixels, box_size_pixels,
                                     edgecolor='red', facecolor='none', linewidth=2, linestyle='--')
            axes[2].add_patch(new_rect)

        # Displaying the updated plot in the Tkinter window
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()  # Remove old canvas

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def show_image(self, image, title):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def find_pattern_in_image(self, binary_mask, min_zero_length=850):
        rows, cols = binary_mask.shape
        for col in range(cols):
            intensity_values = binary_mask[:, col]
            if np.sum(intensity_values == 255) >= 5:
                for row in range(len(intensity_values) - min_zero_length):
                    if intensity_values[row] == 255 and intensity_values[row + 1] == 0 and np.all(intensity_values[row + 1:row + 1 + min_zero_length] == 0):
                        return col, row + 1
        return -1, -1

    def extract_bright_regions(self, binary_mask, row_index_50mm, row_index_5mm_from_50mm, threshold=200):
        roi = binary_mask[row_index_50mm:row_index_5mm_from_50mm, :]
        bright_regions = (roi >= threshold).astype(np.uint8)
        labeled_array, num_features = label(bright_regions)
        objects = find_objects(labeled_array)
        return objects


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
# Paste your full GUI application code here
