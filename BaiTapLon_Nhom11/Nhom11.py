import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        self.original_img = None
        self.processed_img = None
        self.result_img = None
        self.current_img = None
        self.filter_iterations = 0
        self.contrast = 1.0
        self.brightness = 0
        self.current_contrast = 1.0
        self.current_brightness = 0

        # Tạo nút chọn ảnh
        choose_image_button = tk.Button(root, text="Chọn ảnh", command=self.choose_image)
        choose_image_button.pack(pady=20)

        # Tạo nút chức năng
        function_button = tk.Button(root, text="Chức năng", command=self.show_function_window)
        function_button.pack(pady=20)

        # Tạo panel để hiển thị ảnh
        self.panel = tk.Label(self.root)
        self.panel.pack()

    def choose_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_img = cv2.imread(file_path)
            self.processed_img = self.original_img.copy()
            self.result_img = self.original_img.copy()
            self.current_img = self.original_img.copy()
            self.filter_iterations = 0
            self.display_image(self.original_img)

    def apply_median_filter(self):
        if self.processed_img is not None:
            self.result_img = median_filter(self.result_img, filter_size=3)
            self.filter_iterations += 1
            self.current_img = self.result_img.copy()
            self.display_image(self.result_img, text=f"Iteration: {self.filter_iterations}")

    def show_function_window(self):
        function_window = tk.Toplevel(self.root)
        function_window.title("Chức năng")

        # Tạo thanh trượt độ tương phản
        contrast_label = tk.Label(function_window, text="Độ Tương Phản")
        contrast_label.pack()
        contrast_scale = tk.Scale(function_window, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.adjust_contrast)
        contrast_scale.set(self.current_contrast)
        contrast_scale.pack()

        # Tạo thanh trượt độ sáng
        brightness_label = tk.Label(function_window, text="Độ Sáng")
        brightness_label.pack()
        brightness_scale = tk.Scale(function_window, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.adjust_brightness)
        brightness_scale.set(self.current_brightness)
        brightness_scale.pack()

        # Tạo nút áp dụng Median Filter trong cửa sổ chức năng
        apply_median_filter_button = tk.Button(function_window, text="Áp dụng Median Filter", command=self.apply_median_filter)
        apply_median_filter_button.pack(pady=10)

        # Tạo nút zoom ảnh
        zoom_button = tk.Button(function_window, text="Zoom ảnh", command=self.zoom_image)
        zoom_button.pack(pady=10)

        # Tạo nút xoay ảnh
        rotate_button = tk.Button(function_window, text="Xoay ảnh", command=self.rotate_image)
        rotate_button.pack(pady=10)

        # Tạo nút lật trên
        flip_top_button = tk.Button(function_window, text="Lật trên", command=self.flip_top)
        flip_top_button.pack(pady=10)

        # Tạo nút lật dưới
        flip_bottom_button = tk.Button(function_window, text="Lật dưới", command=self.flip_bottom)
        flip_bottom_button.pack(pady=10)

        # Tạo nút lưu kết quả cuối cùng
        save_final_result_button = tk.Button(function_window, text="Lưu kết quả cuối cùng", command=self.save_final_result)
        save_final_result_button.pack(pady=10)

    def zoom_image(self):
        if self.current_img is not None:
            zoom_factor = 1.5
            self.current_img = cv2.resize(self.current_img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
            self.display_image(self.current_img)

    def rotate_image(self):
        if self.current_img is not None:
            rotated_img = cv2.rotate(self.current_img, cv2.ROTATE_90_CLOCKWISE)
            self.current_img = rotated_img
            self.display_image(rotated_img)

    def flip_top(self):
        if self.current_img is not None:
            flipped_img = cv2.flip(self.current_img, 0)
            self.current_img = flipped_img
            self.display_image(flipped_img)

    def flip_bottom(self):
        if self.current_img is not None:
            flipped_img = cv2.flip(self.current_img, 1)
            self.current_img = flipped_img
            self.display_image(flipped_img)

    def adjust_contrast(self, value):
        self.current_contrast = float(value)
        if self.current_img is not None:
            adjusted_img = cv2.convertScaleAbs(self.current_img, alpha=self.current_contrast, beta=self.current_brightness)
            self.display_image(adjusted_img)

    def adjust_brightness(self, value):
        self.current_brightness = int(value)
        if self.current_img is not None:
            adjusted_img = cv2.convertScaleAbs(self.current_img, alpha=self.current_contrast, beta=self.current_brightness)
            self.display_image(adjusted_img)

    def save_final_result(self):
        if self.current_img is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.current_img)
                print(f"Kết quả cuối cùng đã được lưu tại: {file_path}")

    def display_image(self, image, text=""):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.panel.configure(image=img_tk)
        self.panel.image = img_tk

        label = tk.Label(self.root, text=text)
        label.pack()

def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = np.zeros((len(data), len(data[0]), 3), dtype=np.uint8)
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.extend([0, 0, 0])
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.extend([0, 0, 0])
                    else:
                        for k in range(filter_size):
                            temp.extend(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = [temp[len(temp) // 2], temp[len(temp) // 2 + 1], temp[len(temp) // 2 + 2]]
            temp = []

    return data_final

def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
