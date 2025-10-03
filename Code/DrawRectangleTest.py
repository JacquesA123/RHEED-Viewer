import sys
import numpy as np
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets


# ---------- Helper: NumPy 2D uint8 -> QPixmap (grayscale) ----------
def ndarray_to_pixmap(arr: np.ndarray) -> QtGui.QPixmap:
    arr = np.asarray(arr)
    assert arr.ndim == 2, "Expect 2D grayscale"
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    h, w = arr.shape
    qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    qimg = qimg.copy()  # own memory
    return QtGui.QPixmap.fromImage(qimg)


# ---------- Widget to select a rectangular region on a fixed-size QLabel ----------
class RegionSelectLabel(QtWidgets.QLabel):
    regionSelected = QtCore.Signal(QtCore.QRect)

    def __init__(self, pixmap: QtGui.QPixmap, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())          # 1:1 pixels (no scaling)
        self.setMouseTracking(True)
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._origin = None
        self._current_rect = QtCore.QRect()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton:
            self._origin = e.pos()
            self._current_rect = QtCore.QRect(self._origin, QtCore.QSize())
            self._rubber.setGeometry(self._current_rect.normalized())
            self._rubber.show()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._origin is not None:
            self._current_rect = QtCore.QRect(self._origin, e.pos()).normalized()
            self._rubber.setGeometry(self._current_rect)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton and self._origin is not None:
            self._current_rect = QtCore.QRect(self._origin, e.pos()).normalized()
            self._rubber.setGeometry(self._current_rect)
            self.regionSelected.emit(self._current_rect)
            self._origin = None

    def currentRect(self) -> QtCore.QRect:
        return self._current_rect.normalized()


# ---------- Dialog to choose a region ----------
class RegionDialog(QtWidgets.QDialog):
    regionApplied = QtCore.Signal(QtCore.QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Region")
        v = QtWidgets.QVBoxLayout(self)

        # Hard-coded image (256x256): gradient + noise
        base = np.linspace(0, 255, 256, dtype=np.uint8)
        hardcoded = np.tile(base, (256, 1))
        rng = np.random.default_rng(42)
        hardcoded = np.clip(hardcoded + (rng.normal(0, 15, (256, 256))).astype(np.int16), 0, 255).astype(np.uint8)

        pm = ndarray_to_pixmap(hardcoded)
        self.selector = RegionSelectLabel(pm)
        v.addWidget(self.selector, alignment=QtCore.Qt.AlignCenter)

        # Info + buttons
        info = QtWidgets.QLabel("Drag to select a rectangle. Click 'Apply region' to save.")
        v.addWidget(info)

        btns = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply region")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btns.addWidget(self.apply_btn)
        btns.addWidget(self.cancel_btn)
        v.addLayout(btns)

        self.selector.regionSelected.connect(self._on_region_changed)
        self.apply_btn.clicked.connect(self._on_apply)
        self.cancel_btn.clicked.connect(self.reject)

        self._last_rect = QtCore.QRect()

    @QtCore.Slot(QtCore.QRect)
    def _on_region_changed(self, rect: QtCore.QRect):
        self._last_rect = rect

    @QtCore.Slot()
    def _on_apply(self):
        if self._last_rect.isNull() or self._last_rect.width() == 0 or self._last_rect.height() == 0:
            QtWidgets.QMessageBox.information(self, "No selection", "Please drag to select a region first.")
            return
        self.regionApplied.emit(self._last_rect.normalized())
        self.accept()


# ---------- Live random image window ----------
class LiveWindow(QtWidgets.QWidget):
    def __init__(self, selected_rect: QtCore.QRect | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live View")
        self.selected_rect = selected_rect  # in image coords (256x256)

        v = QtWidgets.QVBoxLayout(self)
        v.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)  # window hugs image+text

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)

        self.intensity_label = QtWidgets.QLabel("Intensity: -")
        self.intensity_label.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self.intensity_label)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start()

        self._update_frame()

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.timer.stop()
        super().closeEvent(e)

    @QtCore.Slot()
    def _update_frame(self):
        # 256x256 uint8 random
        frame = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)

        # Draw rectangle overlay if present
        pm = ndarray_to_pixmap(frame)
        if self.selected_rect and not self.selected_rect.isNull():
            p = QtGui.QPainter(pm)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
            pen.setWidth(2)
            p.setPen(pen)
            p.drawRect(self.selected_rect)
            p.end()

            # Compute intensity stats in NumPy using the same rect
            x0 = max(0, self.selected_rect.left())
            y0 = max(0, self.selected_rect.top())
            x1 = min(255, self.selected_rect.right())
            y1 = min(255, self.selected_rect.bottom())
            # Note: QRect is (x,y,w,h); slicing is [row, col] == [y, x]
            region = frame[y0:y1+1, x0:x1+1] if x1 >= x0 and y1 >= y0 else None
            if region is None or region.size == 0:
                text = "Intensity: invalid region"
            else:
                mean_val = float(region.mean())
                sum_val = int(region.sum())
                text = f"Intensity — mean: {mean_val:.2f}, sum: {sum_val}"
            self.intensity_label.setText(text)
        else:
            self.intensity_label.setText("Intensity: (no region)")

        self.image_label.setPixmap(pm)
        self.image_label.resize(pm.size())


# ---------- Main window ----------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Starter")
        v = QtWidgets.QVBoxLayout(self)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_region = QtWidgets.QPushButton("Set region")
        v.addWidget(self.btn_start)
        v.addWidget(self.btn_region)

        self.selected_rect: QtCore.QRect | None = None
        self.live_win: LiveWindow | None = None

        self.btn_region.clicked.connect(self.on_set_region)
        self.btn_start.clicked.connect(self.on_start)

    @QtCore.Slot()
    def on_set_region(self):
        dlg = RegionDialog(self)
        dlg.regionApplied.connect(self._on_region_applied)
        dlg.exec()

    @QtCore.Slot(QtCore.QRect)
    def _on_region_applied(self, rect: QtCore.QRect):
        self.selected_rect = rect
        # Optionally, show feedback
        QtWidgets.QMessageBox.information(
            self, "Region set",
            f"Rect: x={rect.x()}, y={rect.y()}, w={rect.width()}, h={rect.height()}"
        )

    @QtCore.Slot()
    def on_start(self):
        # Create or show the live window with the stored region
        if self.live_win is None or not self.live_win.isVisible():
            # ✨ No parent → top-level window
            self.live_win = LiveWindow(self.selected_rect)  
            # (optional) ensure it behaves like a standalone window
            self.live_win.setWindowFlag(QtCore.Qt.Window, True)
            self.live_win.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        else:
            # Update its region if already open
            self.live_win.selected_rect = self.selected_rect

        self.live_win.show()
        self.live_win.raise_()
        self.live_win.activateWindow()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
