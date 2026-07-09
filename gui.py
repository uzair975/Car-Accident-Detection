import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from app import run_image
from config import ACCIDENT_MODEL_PATHS

MAX_PREVIEW_WIDTH = 900
MAX_PREVIEW_HEIGHT = 600


def _bgr_to_rgb_image(bgr_image):
    rgb_image = bgr_image[:, :, ::-1]
    return Image.fromarray(rgb_image)


def _resize_image(pil_image, max_width, max_height):
    width, height = pil_image.size
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        return pil_image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return pil_image


class AccidentDetectorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Accident Detector")
        self.geometry("1200x820")
        self.resizable(True, True)

        self.image_path = None
        self.photo_image = None

        self._create_widgets()

    def _create_widgets(self):
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=10)

        ttk.Button(controls, text="Select Image", command=self._select_image).pack(side="left")

        self.image_path_label = ttk.Label(controls, text="No image selected", wraplength=620)
        self.image_path_label.pack(side="left", padx=10)

        self.run_button = ttk.Button(controls, text="Detect Accident", command=self._run_detection)
        self.run_button.pack(side="right")

        default_model = next((path for path in ACCIDENT_MODEL_PATHS if path.exists()), None)
        model_text = f"Accident model: {default_model}" if default_model else "Accident model: not found"

        self.model_path_label = ttk.Label(
            controls,
            text=model_text,
            wraplength=280,
            justify="right",
        )
        self.model_path_label.pack(side="right", padx=10)

        content = ttk.PanedWindow(self, orient="horizontal")
        content.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left_frame = ttk.Frame(content)
        right_frame = ttk.Frame(content, width=380)
        content.add(left_frame, weight=3)
        content.add(right_frame, weight=1)

        self.canvas = tk.Canvas(left_frame, bg="#222222")
        self.canvas.pack(fill="both", expand=True)

        self.result_text = tk.Text(right_frame, wrap="word", state="disabled", width=50)
        self.result_text.pack(fill="both", expand=True, padx=(5, 0), pady=0)
        self.result_text.tag_configure("title", font=("TkDefaultFont", 10, "bold"))

        self.status_label = ttk.Label(self, text="Ready", anchor="w")
        self.status_label.pack(fill="x", padx=10, pady=(0, 10))

    def _select_image(self):
        selected = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*")] ,
        )
        if not selected:
            return

        self.image_path = Path(selected)
        self.image_path_label.config(text=str(self.image_path))
        self._display_image_preview(self.image_path)

    def _display_image_preview(self, image_path: Path):
        image = Image.open(image_path)
        preview = _resize_image(image, MAX_PREVIEW_WIDTH, MAX_PREVIEW_HEIGHT)
        self.photo_image = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _run_detection(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Please select an image before running detection.")
            return

        self.run_button.config(state="disabled")
        self.status_label.config(text="Running detection... this may take a while")
        threading.Thread(target=self._detect_in_background, daemon=True).start()

    def _detect_in_background(self):
        try:
            report = run_image(self.image_path, output_path=None, report_path=None, accident_model=None)
            self.after(0, self._display_results, report)
        except Exception as exc:
            self.after(0, self._show_error, exc)
        finally:
            self.after(0, self._run_finished)

    def _display_results(self, report: dict):
        image = self._render_annotated_image(report)
        self._display_tk_image(image)

        content = json.dumps(report, indent=2)
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", content)
        self.result_text.config(state="disabled")

        status_text = (
            f"Accident detected: {report['accident_detected']} | "
            f"Probability: {report['accident_probability']:.2f} | "
            f"Explanation: {report.get('explanation', 'N/A')}"
        )
        self.status_label.config(text=status_text)

    def _render_annotated_image(self, report: dict):
        from visualize import draw_results

        import cv2

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")

        annotated = draw_results(image, report)
        return _bgr_to_rgb_image(annotated)

    def _display_tk_image(self, pil_image: Image.Image):
        preview = _resize_image(pil_image, MAX_PREVIEW_WIDTH, MAX_PREVIEW_HEIGHT)
        self.photo_image = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _show_error(self, error: Exception):
        messagebox.showerror("Detection error", f"An error occurred:\n{error}")
        self.status_label.config(text="Error during detection")

    def _run_finished(self):
        self.run_button.config(state="normal")


def main():
    app = AccidentDetectorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
