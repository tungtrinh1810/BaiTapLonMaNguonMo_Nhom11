import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import threading
from collections import deque

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Giao dien xu li anh")

        self.original_img = None
        self.processed_img = None
        self.result_img = None
        self.current_img = None
        self.filter_iterations = 0
        self.contrast = 1.0
        self.brightness = 0
        self.angle = 0
        self.lock = threading.Lock()

        self.undo_stack = deque()
        self.redo_stack = deque()

        choose_image_button = tk.Button(root, text="Chọn ảnh", command=self.choose_image)
        choose_image_button.pack(pady=20)

        apply_median_filter_button = tk.Button(root, text="Áp dụng Median Filter", command=self.apply_median_filter)
        apply_median_filter_button.pack(pady=20)

        function_button = tk.Button(root, text="Chức năng", command=self.show_function_window)
        function_button.pack(pady=20)

        undo_button = tk.Button(root, text="Undo", command=self.undo)
        undo_button.pack(side=tk.LEFT, padx=10)

        redo_button = tk.Button(root, text="Redo", command=self.redo)
        redo_button.pack(side=tk.RIGHT, padx=10)

        self.panel = tk.Label(self.root)
        self.panel.pack()

    def choose_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.original_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if self.original_img is not None:
                    self.processed_img = self.original_img.copy()
                    self.result_img = self.original_img.copy()
                    self.current_img = self.original_img.copy()
                    self.filter_iterations = 0
                    self.clear_undo_redo_stack()
                    self.display_image(self.original_img)
                else:
                    messagebox.showinfo("Thông báo", "Không thể đọc được ảnh. Vui lòng chọn một file hỗ trợ.")
            except Exception as e:
                messagebox.showinfo("Thông báo", f"Đã xảy ra lỗi: {str(e)}")

    def apply_median_filter(self):
        if self.processed_img is not None:
            if len(self.processed_img.shape) > 2:
                messagebox.showinfo("Thông báo", "Vui lòng nhập ảnh xám trước khi áp dụng Median Filter cho ảnh màu.")
                return

            with self.lock:
                self.push_to_undo_stack(self.result_img.copy())
                self.result_img = self.median_filter(self.result_img, filter_size=3)
                self.filter_iterations += 1
                self.current_img = self.result_img.copy()
                self.display_image(self.result_img, text=f"Iteration: {self.filter_iterations}")
        else:
            messagebox.showinfo("Thông báo", "Vui lòng chọn ảnh trước khi áp dụng Median Filter.")

    def show_function_window(self):
        if self.original_img is not None:
            function_window = tk.Toplevel(self.root)
            function_window.title("Chức năng")

            rotate_button = tk.Button(function_window, text="Xoay ảnh", command=self.rotate_image)
            rotate_button.pack(pady=10)

            flip_top_button = tk.Button(function_window, text="Lật trên dưới", command=self.flip_top)
            flip_top_button.pack(pady=10)

            flip_bottom_button = tk.Button(function_window, text="Lật trái phải", command=self.flip_bottom)
            flip_bottom_button.pack(pady=10)

            contrast_label = tk.Label(function_window, text="Độ Tương Phản")
            contrast_label.pack()
            contrast_scale = tk.Scale(function_window, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.adjust_contrast)
            contrast_scale.set(self.contrast)
            contrast_scale.pack()

            brightness_label = tk.Label(function_window, text="Độ Sáng")
            brightness_label.pack()
            brightness_scale = tk.Scale(function_window, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.adjust_brightness)
            brightness_scale.set(self.brightness)
            brightness_scale.pack()

            save_final_result_button = tk.Button(function_window, text="Lưu kết quả cuối cùng", command=self.save_final_result)
            save_final_result_button.pack(pady=10)

            blur_all_button = tk.Button(function_window, text="Làm mờ toàn bộ ảnh", command=self.blur_entire_image)
            blur_all_button.pack(pady=10)

            denoise_button = tk.Button(function_window, text="Làm mịn ảnh", command=self.denoise_image)
            denoise_button.pack(pady=10)
        else:
            messagebox.showinfo("Thông báo", "Vui lòng chọn ảnh trước khi sử dụng chức năng.")

    def denoise_image(self):
        if self.result_img is not None:
            with self.lock:
                self.push_to_undo_stack(self.result_img.copy())
                if len(self.result_img.shape) == 2:  # Grayscale image
                    denoised_img = cv2.fastNlMeansDenoising(self.result_img, None, 10, 7, 21)
                else:  # Colored image
                    denoised_img = cv2.fastNlMeansDenoisingColored(self.result_img, None, 10, 10, 7, 21)
                self.result_img = denoised_img.copy()
                self.current_img = self.result_img.copy()
                self.display_image(denoised_img, text="Ảnh đã được lọc nhiễu")

    def rotate_image(self):
        if self.result_img is not None:
            with self.lock:
                self.push_to_undo_stack(self.result_img.copy())
                if len(self.result_img.shape) == 3:  # Ảnh màu (3 kênh)
                    h, w, _ = self.result_img.shape
                else:  # Ảnh xám (2 kênh)
                    h, w = self.result_img.shape
                self.angle = (self.angle + 90) % 360
                M = cv2.getRotationMatrix2D((w // 2, h // 2), self.angle, 1)
                rotated_img = cv2.warpAffine(self.result_img, M, (w, h), flags=cv2.INTER_LINEAR)
                self.result_img = rotated_img.copy()
                self.current_img = self.result_img.copy()
                self.display_image(rotated_img)

    def flip_top(self):
        if self.result_img is not None:
            with self.lock:
                self.push_to_undo_stack(self.result_img.copy())
                flipped_img = cv2.flip(self.current_img, 0)
                self.result_img = flipped_img.copy()
                self.current_img = self.result_img.copy()
                self.display_image(flipped_img)

    def flip_bottom(self):
        if self.result_img is not None:
            with self.lock:
                self.push_to_undo_stack(self.result_img.copy())
                flipped_img = cv2.flip(self.current_img, 1)
                self.result_img = flipped_img.copy()
                self.current_img = self.result_img.copy()
                self.display_image(flipped_img)

    def adjust_contrast(self, value):
        self.contrast = float(value)
        if self.current_img is not None:
            with self.lock:
                self.push_to_undo_stack(self.current_img.copy())
                adjusted_img = cv2.convertScaleAbs(self.current_img, alpha=self.contrast, beta=self.brightness)
                self.result_img = adjusted_img.copy()
                self.display_image(adjusted_img)

    def adjust_brightness(self, value):
        self.brightness = int(value)
        if self.current_img is not None:
            with self.lock:
                self.push_to_undo_stack(self.current_img.copy())
                adjusted_img = cv2.convertScaleAbs(self.current_img, alpha=self.contrast, beta=self.brightness)
                self.result_img = adjusted_img.copy()
                self.display_image(adjusted_img)

    def save_final_result(self):
        if self.current_img is not None:
            with self.lock:
                self.push_to_undo_stack(self.current_img.copy())
                file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
                if file_path:
                    adjusted_img = cv2.convertScaleAbs(self.current_img, alpha=self.contrast, beta=self.brightness)
                    cv2.imwrite(file_path, adjusted_img)
                    print(f"Kết quả cuối cùng đã được lưu tại: {file_path}")

    def blur_entire_image(self):
        if self.result_img is not None:
            with self.lock:
                self.push_to_undo_stack(self.result_img.copy())
                blurred_img = cv2.GaussianBlur(self.result_img, (15, 15), 0)
                self.result_img = blurred_img.copy()
                self.current_img = self.result_img.copy()
                self.display_image(blurred_img, text="Ảnh đã được làm mờ toàn bộ")

    def display_image(self, image, text=""):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.panel.configure(image=img_tk)
        self.panel.image = img_tk

        label = tk.Label(self.root, text=text)
        label.pack()

    def push_to_undo_stack(self, img):
        self.undo_stack.append(img)
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            with self.lock:
                img = self.undo_stack.pop()
                self.redo_stack.append(self.current_img.copy())
                self.current_img = img.copy()
                self.display_image(img)

    def redo(self):
        if self.redo_stack:
            with self.lock:
                img = self.redo_stack.pop()
                self.undo_stack.append(self.current_img.copy())
                self.current_img = img.copy()
                self.display_image(img)

    def clear_undo_redo_stack(self):
        self.undo_stack.clear()
        self.redo_stack.clear()

    def median_filter(self, data, filter_size):
        indexer = filter_size // 2
        data_final = np.zeros_like(data, dtype=np.uint8)

        for i in range(len(data)):
            for j in range(len(data[0])):
                temp = []
                for z in range(filter_size):
                    for k in range(filter_size):
                        row_idx = i + z - indexer
                        col_idx = j + k - indexer
                        if 0 <= row_idx < len(data) and 0 <= col_idx < len(data[0]):
                            temp.append(data[row_idx][col_idx])

                temp.sort()
                median_pixel = temp[len(temp) // 2]
                data_final[i][j] = median_pixel

        return data_final

def main():
    try:
        root = tk.Tk()
        app = ImageProcessingApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")

if __name__ == "__main__":
    main()
