import sys
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

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)

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

        # ----- Convert to QPixmap -----
        bytes_per_line = 3 * w
        qimage = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        qimage = qimage.copy()  # keep buffer alive
        pixmap = QtGui.QPixmap.fromImage(qimage)

        # ----- Display -----
        self.label.setPixmap(pixmap.scaled(
            self.label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))
        self.label.setStyleSheet("")
        self.label.setText("")
        self.setWindowTitle(f"Viewing: {path}")

    def resizeEvent(self, event):
        # Rescale image when window resizes
        if not self.label.pixmap():
            return
        self.label.setPixmap(self.label.pixmap().scaled(
            self.label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))
        super().resizeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = NpyDropViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
