import sys
from typing import Optional

import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore


class NpyDropViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NumPy Image Viewer (Drag .npy Here)")
        self.setAcceptDrops(True)
        self.resize(800, 600)

        self.label = QtWidgets.QLabel("Drag and drop a .npy image file here")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 16px; color: gray;")

        self.current_pixmap: Optional[QtGui.QPixmap] = None
        self.original_rgb: Optional[np.ndarray] = None

        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)  # 10% – 300%
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._on_adjustment_change)

        self.exposure_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exposure_slider.setRange(-100, 100)  # -1.0 – +1.0 stops
        self.exposure_slider.setValue(0)
        self.exposure_slider.valueChanged.connect(self._on_adjustment_change)

        self.contrast_value_label = QtWidgets.QLabel("100%")
        self.exposure_value_label = QtWidgets.QLabel("0.00")

        controls_layout = QtWidgets.QFormLayout()
        contrast_row = QtWidgets.QHBoxLayout()
        contrast_row.setContentsMargins(0, 0, 0, 0)
        contrast_row.addWidget(self.contrast_slider)
        contrast_row.addWidget(self.contrast_value_label)
        contrast_container = QtWidgets.QWidget()
        contrast_container.setLayout(contrast_row)
        controls_layout.addRow("Contrast", contrast_container)

        exposure_row = QtWidgets.QHBoxLayout()
        exposure_row.setContentsMargins(0, 0, 0, 0)
        exposure_row.addWidget(self.exposure_slider)
        exposure_row.addWidget(self.exposure_value_label)
        exposure_container = QtWidgets.QWidget()
        exposure_container.setLayout(exposure_row)
        controls_layout.addRow("Exposure", exposure_container)

        self.controls_widget = QtWidgets.QWidget()
        self.controls_widget.setLayout(controls_layout)
        self.controls_widget.setEnabled(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.controls_widget)

    # ---------- Drag and drop ----------
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if not path.lower().endswith(".npy"):
            QtWidgets.QMessageBox.warning(self, "Invalid file", "Please drop a .npy file.")
            return
        try:
            arr = np.load(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Could not load file:\n{e}")
            return

        self.display_numpy_image(arr, path)

    # ---------- Display logic ----------
    def display_numpy_image(self, arr: np.ndarray, path: str):
        arr = np.asarray(arr)
        arr = np.squeeze(arr)  # handles (H,W,1) → (H,W)

        # Normalize non-uint8
        if arr.dtype != np.uint8:
            a_min, a_max = float(arr.min()), float(arr.max())
            if a_max <= a_min:
                a_max = a_min + 1.0
            arr = ((arr - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)

        # ----- Interpret shape -----
        if arr.ndim == 2:
            # Grayscale → convert to RGB for consistent display
            h, w = arr.shape
            rgb = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] == 3:
            # (3,H,W)
            rgb = np.moveaxis(arr, 0, 2)
            h, w, _ = rgb.shape
        elif arr.ndim == 3 and arr.shape[2] == 1:
            # (H,W,1)
            rgb = np.concatenate([arr] * 3, axis=2)
            h, w, _ = rgb.shape
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # (H,W,3)
            rgb = arr
            h, w, _ = rgb.shape
        else:
            QtWidgets.QMessageBox.warning(self, "Unsupported Shape",
                                          f"Cannot display array with shape {arr.shape}")
            return

        # Store as float for adjustments
        self.original_rgb = rgb.astype(np.float32) / 255.0

        # Reset controls to defaults for new image
        self._reset_adjustments()
        self.controls_widget.setEnabled(True)

        # Update title and display with current adjustments
        self.setWindowTitle(f"Viewing: {path}")
        self._apply_adjustments_and_display()

    def resizeEvent(self, event):
        # Rescale image when window resizes
        self._update_label_pixmap()
        super().resizeEvent(event)

    # ---------- Adjustment helpers ----------
    def _reset_adjustments(self):
        # Block signals to avoid duplicate updates while resetting values
        self.contrast_slider.blockSignals(True)
        self.exposure_slider.blockSignals(True)
        self.contrast_slider.setValue(100)
        self.exposure_slider.setValue(0)
        self.contrast_slider.blockSignals(False)
        self.exposure_slider.blockSignals(False)
        self._update_adjustment_labels()

    def _on_adjustment_change(self):
        if self.original_rgb is None:
            return
        self._update_adjustment_labels()
        self._apply_adjustments_and_display()

    def _update_adjustment_labels(self):
        contrast_display = f"{self.contrast_slider.value()}%"
        exposure_display = f"{self.exposure_slider.value() / 100:.2f}"
        self.contrast_value_label.setText(contrast_display)
        self.exposure_value_label.setText(exposure_display)

    def _apply_adjustments_and_display(self):
        if self.original_rgb is None:
            return

        contrast = self.contrast_slider.value() / 100.0
        exposure = self.exposure_slider.value() / 100.0

        adjusted = (self.original_rgb - 0.5) * contrast + 0.5 + exposure
        adjusted = np.clip(adjusted, 0.0, 1.0)
        uint8_img = (adjusted * 255.0).astype(np.uint8)
        h, w, _ = uint8_img.shape

        bytes_per_line = 3 * w
        qimage = QtGui.QImage(uint8_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        qimage = qimage.copy()  # keep buffer alive
        self.current_pixmap = QtGui.QPixmap.fromImage(qimage)

        self.label.setStyleSheet("")
        self.label.setText("")
        self._update_label_pixmap()

    def _update_label_pixmap(self):
        if not self.current_pixmap:
            return
        self.label.setPixmap(self.current_pixmap.scaled(
            self.label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = NpyDropViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
