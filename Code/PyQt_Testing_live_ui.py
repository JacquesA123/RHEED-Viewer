# Revised PySide6 GUI for RHEED live view + acquisition/stream control
# - Initial screen shows only "Start Live Feed"
# - Once live starts: shows "Stop Live Feed", "Acquire RHEED Image", "Start RHEED Stream"
# - Live view DOES NOT save images; it just displays frames continuously
# - "Start RHEED Stream" saves frames at a chosen frequency for a chosen duration, while continuing to update the live view
# - "Acquire RHEED Image" saves a single frame
# - Camera access is coordinated so only one stream owns the camera at a time

import sys
import time
import os
import datetime
import threading
import plotly.express as px
import pandas as pd
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from vmbpy import VmbSystem, Camera, Stream, Frame
from layout_colorwidget import Color
from PyrometerControl import get_pyrometer_temperature, start_pyrometer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from GetOscillationPlot import obtain_pixmap
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from GetOscillationPlot import plotly_figure_to_qpixmap
import pyqtgraph



# Image saving locations
# single_images_folder = os.path.expanduser(r"C:\Jacques RHEED\RHEED Images\Single Images") # Oxide MBE
# stream_images_folder = os.path.expanduser(r"C:\Jacques RHEED\RHEED Images\Stream Images") # Oxide MBE
single_images_folder = os.path.expanduser(r"C:\Dropbox\Data\RHEED\RHEED_YangGroup\FeSeTe_STO\10_03_2025")

stream_images_folder = os.path.expanduser(r"C:\Dropbox\Data\RHEED\RHEED_YangGroup\FeSeTe_STO\10_03_2025")

os.makedirs(single_images_folder, exist_ok=True)
os.makedirs(stream_images_folder, exist_ok=True)



# --------------------------------------------------------------------------------------
# Utility: convert numpy 2D array (grayscale) to QPixmap
# --------------------------------------------------------------------------------------
def ndarray_to_pixmap(arr: np.ndarray) -> QtGui.QPixmap:
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.dtype != np.uint8:
        # Normalize to 8-bit if needed
        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max <= a_min:
            a_max = a_min + 1.0
        arr = ((arr - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)

    h, w = arr.shape
    bytes_per_line = w
    qimage = QtGui.QImage(arr.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
    # Important: keep a copy so the data isn't freed after function returns
    qimage = qimage.copy()
    return QtGui.QPixmap.fromImage(qimage)


# --------------------------------------------------------------------------------------
# Dialog for RHEED stream settings
# --------------------------------------------------------------------------------------
class StreamSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stream Preferences")
        layout = QtWidgets.QFormLayout(self)

        self.duration_input = QtWidgets.QLineEdit(self)
        self.duration_input.setPlaceholderText("Enter stream duration in seconds")

        self.frequency_input = QtWidgets.QLineEdit(self)
        self.frequency_input.setPlaceholderText("Enter image frequency in Hz")

        layout.addRow("Stream Duration (seconds):", self.duration_input)
        layout.addRow("Image Frequency (Hz):", self.frequency_input)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        try:
            duration = float(self.duration_input.text())
            frequency = float(self.frequency_input.text())
            return duration, frequency
        except ValueError:
            return None, None

# --------------------------------------------------------------------------------------
# Window that shows the live RHEED feed (grayscale image)
# --------------------------------------------------------------------------------------


class LiveImageWindow(QtWidgets.QWidget):
    def __init__(self, pyrometer_app, intensity_rectangle: QtCore.QRect | None = None, title="Live RHEED Feed", parent=None):
        super().__init__(parent)
        self.pyrometer_app = pyrometer_app
        if intensity_rectangle is not None and intensity_rectangle.isNull():
            intensity_rectangle = None
        self._intensity_rect: QtCore.QRect | None = intensity_rectangle.normalized() if intensity_rectangle is not None else None
        self.setWindowTitle(title)
        self.resize(1600, 1200)  # use resize instead of setGeometry for a top-level

        # --- Layout: create it WITHOUT a parent, then set once ---
        layout = QtWidgets.QVBoxLayout()
        # layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(8)

        # --- Image area ---
        self.image_label = QtWidgets.QLabel()
        # self.test_image = QtWidgets.QLabel()
        # self.test_image.setPixmap(QtGui.QPixmap(r"C:\Users\Lab10\Pictures\Screenshots\Screenshot (1).png"))
        layout.addWidget(self.image_label)
        # layout.addWidget(self.test_image)

        # --- Live intensity plot ---
        self.plot_widget = pyqtgraph.PlotWidget()
        self.plot_widget.setLabel("left", "Intensity")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.intensity_curve = self.plot_widget.plot(pen=pyqtgraph.mkPen(color="y", width=2))
        layout.addWidget(self.plot_widget)

        # Data containers for the intensity time series
        self._intensity_time = []
        self._intensity_values = []
        self._start_time = time.monotonic()




        self.setLayout(layout)









    def set_intensity_rectangle(self, rect: QtCore.QRect | None):
        """Update the rectangle used for intensity calculations and reset the plot."""
        if rect is not None:
            rect = rect.normalized()
            if rect.isNull() or rect.width() <= 0 or rect.height() <= 0:
                rect = None
        self._intensity_rect = rect
        self.reset_intensity_plot()

    @QtCore.Slot(np.ndarray)
    def update_image(self, image_array: np.ndarray):

        image_array = np.asarray(image_array)
        pix = ndarray_to_pixmap(image_array)

        # --- draw timestamp (top-left) on the pixmap ---
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temperature = get_pyrometer_temperature(self.pyrometer_app)


        time_and_temperature_text = ts + '  ' + temperature + 'C'
        p = QtGui.QPainter(pix) # Creates a QPainter bound to the pixmap, so you can draw text or graphics onto the pixmap
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)

        # Pixmap text overlay settings
        font = QtGui.QFont("DejaVu Sans Mono", 12)
        font.setStyleHint(QtGui.QFont.TypeWriter)
        p.setFont(font)

        # Creates a rectangle for the text to reside in
        metrics = QtGui.QFontMetrics(font)
        bg_rect = metrics.boundingRect(time_and_temperature_text).adjusted(-6, -4, +6, +4)
        bg_rect.moveTopLeft(QtCore.QPoint(5, 5))  # small margin from top-left

        # semi-transparent background in the rectangle for readability
        p.fillRect(bg_rect, QtGui.QColor(0, 0, 0, 160))
        p.setPen(QtCore.Qt.white)
        p.drawText(bg_rect.left() + 6, bg_rect.top() + 4 + metrics.ascent(), time_and_temperature_text)
        p.end() # Finalizes the drawing operations on the pixmap
        

        rect = self._intensity_rect
        intensity: float | None = None

        if rect is not None:
            painter = QtGui.QPainter(pix)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.end()

            height, width = image_array.shape[0], image_array.shape[1]
            x0 = max(0, rect.x())
            y0 = max(0, rect.y())
            x1 = min(width, rect.x() + rect.width())
            y1 = min(height, rect.y() + rect.height())

            if x1 > x0 and y1 > y0:
                sub_rectangle = image_array[y0:y1, x0:x1]
                intensity = float(np.asarray(sub_rectangle, dtype=np.float64).sum())
                print(f'intensity = {intensity}')

        self.image_label.setPixmap(pix) # Update the QLabel widget with the new pixmap (image)
        self.image_label.resize(pix.size()) # Resize the label so the window fits the pixmap size exactly

        if intensity is not None:
            elapsed = time.monotonic() - self._start_time
            self._intensity_time.append(elapsed)
            self._intensity_values.append(intensity)

            max_points = 15000
            if len(self._intensity_time) > max_points:
                self._intensity_time.pop(0)
                self._intensity_values.pop(0)

            self.intensity_curve.setData(self._intensity_time, self._intensity_values)

    def reset_intensity_plot(self):
        """Clear the stored intensity history and restart the timer baseline."""
        self._intensity_time.clear()
        self._intensity_values.clear()
        self._start_time = time.monotonic()
        self.intensity_curve.clear()

    @QtCore.Slot(Figure)
    def figure_to_pixmap(fig: Figure) -> QtGui.QPixmap:
        """Render a Matplotlib Figure to QPixmap."""
        canvas = FigureCanvas(fig)             # temporary canvas
        print('initialized canvas')
        canvas.draw()
        print('drew canvas')
        buf, (w, h) = canvas.print_to_buffer()
        qimage = QtGui.QImage(
            buf, w, h, QtGui.QImage.Format_RGBA8888
        )
        print('initialized qimage')
        return QtGui.QPixmap.fromImage(qimage)

        


# --------------------------------------------------------------------------------------
# Base class for camera threads
# --------------------------------------------------------------------------------------
class _BaseCameraThread(QtCore.QThread):
    new_frame = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None

    def stop(self):
        with self._lock:
            self._running = False

    def _set_running(self, val: bool):
        with self._lock:
            self._running = val

    def _is_running(self) -> bool:
        with self._lock:
            return self._running

    # Set from handler so single-acquire can also save latest frame if needed
    def _update_latest_frame(self, frame_np: np.ndarray):
        with self._lock:
            self._latest_frame = frame_np

    def latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()
 
class SelectRegionReferenceFileTemplate(QtWidgets.QDialog):
    """Dialog where the user pastes a file path into a textbox and clicks OK.
       After that, opens RegionDialog to select a rectangle."""
    regionSelected = QtCore.Signal(QtCore.QRect, str)  # emits (rect, image_path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Reference Image Path")
        self.selected_region = None
        self.selected_image_path = None

        layout = QtWidgets.QVBoxLayout(self)

        # Label + textbox for file path
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Paste image file path here...")
        layout.addWidget(self.path_edit)

        # Buttons
        btns = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btns.addWidget(self.ok_btn)
        btns.addWidget(self.cancel_btn)
        layout.addLayout(btns)

        # Connect signals
        self.ok_btn.clicked.connect(self.on_ok_clicked)
        self.cancel_btn.clicked.connect(self.reject)

    @QtCore.Slot()
    def on_ok_clicked(self):
        file_path = self.path_edit.text().strip()
        file_path = file_path.strip('"').strip("'")
        if not file_path:
            QtWidgets.QMessageBox.warning(self, "No Path", "Please paste a file path first.")
            return
        if not QtCore.QFile.exists(file_path):
            QtWidgets.QMessageBox.critical(self, "Invalid Path", f"File not found:\n{file_path}")
            return

        # Launch region dialog with the provided path
        region_dialog = RegionDialog(file_path, self)
        if region_dialog.exec() == QtWidgets.QDialog.Accepted:
            rect = region_dialog.get_selected_rect()
            if rect is not None and not rect.isNull():
                self.selected_region = rect
                self.selected_image_path = file_path
                self.regionSelected.emit(rect, file_path)
                print("User selected region:", rect, "on", file_path)
            self.accept()


# ---------- Dialog to set the oscillation region (uses image path) ----------
class RegionDialog(QtWidgets.QDialog):
    regionApplied = QtCore.Signal(QtCore.QRect)

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Region")
        self._last_rect = QtCore.QRect()

        v = QtWidgets.QVBoxLayout(self)

        # Load the chosen image from disk
        numpy_image = np.load(image_path)
        converted_image = ndarray_to_pixmap(numpy_image)
        pm = QtGui.QPixmap(converted_image)
        if pm.isNull():
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not load image:\n{image_path}")
            self.reject()
            return

        # 1:1 pixels selector
        self.selector = RegionSelectLabel(pm)
        v.addWidget(self.selector, alignment=QtCore.Qt.AlignCenter)

        # Info + buttons
        v.addWidget(QtWidgets.QLabel("Drag to select a rectangle. Click 'Apply region' to save."))

        btns = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply region")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btns.addWidget(self.apply_btn)
        btns.addWidget(self.cancel_btn)
        v.addLayout(btns)

        # Signals
        self.selector.regionSelected.connect(self._on_region_changed)
        self.apply_btn.clicked.connect(self._on_apply)
        self.cancel_btn.clicked.connect(self.reject)

    @QtCore.Slot(QtCore.QRect)
    def _on_region_changed(self, rect: QtCore.QRect):
        self._last_rect = rect.normalized()

    @QtCore.Slot()
    def _on_apply(self):
        if self._last_rect.isNull() or self._last_rect.width() == 0 or self._last_rect.height() == 0:
            QtWidgets.QMessageBox.information(self, "No selection", "Please drag to select a region first.")
            return
        self.regionApplied.emit(self._last_rect)
        self.accept()

    def get_selected_rect(self) -> QtCore.QRect | None:
        return self._last_rect if not self._last_rect.isNull() else None


# ---------- Widget to select a rectangular region on a fixed-size QLabel ----------
class RegionSelectLabel(QtWidgets.QLabel):
    regionSelected = QtCore.Signal(QtCore.QRect)

    def __init__(self, pixmap: QtGui.QPixmap, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())  # 1:1 pixels (no scaling)
        self.setMouseTracking(True)
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._origin = None
        self._current_rect = QtCore.QRect()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton:
            self._origin = e.pos()
            self._current_rect = QtCore.QRect(self._origin, QtCore.QSize())
            self._rubber.setGeometry(self._current_rect)
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


# --------------------------------------------------------------------------------------
# Live view thread: displays frames continuously, DOES NOT SAVE
# --------------------------------------------------------------------------------------
class LiveViewWorker(_BaseCameraThread):
    def __init__(self, target_hz: float):
        super().__init__()
        print(float(target_hz))
        self.target_hz = float(target_hz)

    def run(self):
        self._set_running(True)

        

        # Real camera path using vmbpy (Allied Vision)
        with VmbSystem.get_instance() as vmb:
            print('opening live view camera')
            cams = vmb.get_all_cameras()
            print(cams[0])
            if not cams:
                return
            cam = cams[0]
            with cam:
                # Configure triggering
                try:
                    cam.TriggerSource.set('Software')
                    cam.TriggerSelector.set('FrameStart')
                    cam.TriggerMode.set('On')
                    cam.AcquisitionMode.set('Continuous')
                except Exception:
                    pass

                # Frame callback
                def _handler(c: Camera, s: Stream, f: Frame):
                    c.queue_frame(f)
                    
                    img = f.as_numpy_ndarray()
                    # print(img.dtype)
                    if img.dtype != np.uint8:
                        # Normalize to 8-bit if needed
                        a_min, a_max = float(img.min()), float(img.max())
                        if a_max <= a_min:
                            a_max = a_min + 1.0
                        img = ((img - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)
                    self._update_latest_frame(img)
                    self.new_frame.emit(img)

                cam.start_streaming(_handler)
                period = 1.0 / max(self.target_hz, 1.0)

                try:
                    while self._is_running():
                        try:
                            cam.TriggerSoftware.run()
                        except Exception:
                            pass
                        time.sleep(period)
                finally:
                    cam.stop_streaming()


            print('closing live view camera')

    


# --------------------------------------------------------------------------------------
# Stream-with-save thread: saves frames at given frequency for duration; keeps preview updated
# --------------------------------------------------------------------------------------
class SaveStreamWorker(_BaseCameraThread):
    def __init__(self, pyrometer_app, duration_s: float, freq_hz: float, out_dir: str):
        super().__init__()
        self.duration_s = float(duration_s)
        self.pyrometer_app = pyrometer_app
        self.freq_hz = float(freq_hz)
        self.out_dir = out_dir

    def run(self):
        self._set_running(True)
        os.makedirs(self.out_dir, exist_ok=True)

        

        with VmbSystem.get_instance() as vmb:
            print('opening acquisition stream camera')
            cams = vmb.get_all_cameras()
            if not cams:
                print("No camera found")
                return
            cam = cams[0]

            with cam:
                try:
                    cam.TriggerSource.set('Software')
                    cam.TriggerSelector.set('FrameStart')
                    cam.TriggerMode.set('On')
                    cam.AcquisitionMode.set('Continuous')
                    print('Camera parameters set')
                except Exception as e:
                    print(f'Parameter setup failed: {e}')

                try:
                    cam.PixelFormat.set('Mono8')
                    print('Pixel format set')
                except Exception:
                    pass

                def _handler(c: Camera, s: Stream, f: Frame):
                    print('Frame acquired: {}'.format(f), flush=True)
                    c.queue_frame(f)
                    img = f.as_numpy_ndarray()
                    if img.dtype != np.uint8:
                        # Normalize to 8-bit if needed
                        a_min, a_max = float(img.min()), float(img.max())
                        if a_max <= a_min:
                            a_max = a_min + 1.0
                        img = ((img - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)
                    # print(img.dtype)
                    self._save_frame(img)      # save immediately
                    self.new_frame.emit(img)   # update GUI

                cam.start_streaming(_handler)
                print('Streaming started')
                # QtCore.QThread.msleep(5)  # warmup

                trigger_period = 1.0 / self.freq_hz
                next_trigger_t = time.time()
                end_time = time.time() + self.duration_s
                

                try:
                    while self._is_running() and time.time() < end_time:
                        now = time.time()
                        if now >= next_trigger_t:
                            try:
                                cam.TriggerSoftware.run()
                                # print('Trigger fired', flush=True)
                            except Exception as e:
                                print(f'trigger error: {e}', flush=True)
                            next_trigger_t = now + trigger_period
                            QtCore.QThread.msleep(int(trigger_period * 1000))
                finally:
                    cam.stop_streaming()
            print('closing acquisition stream camera')


    def _save_frame(self, frame_np: np.ndarray):
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        

        temperature = get_pyrometer_temperature(self.pyrometer_app)
        # temperature = 'geo'
        # print(type(temperature))
        # print(f'temperature = {temperature}')

        stream_image_name = temperature + '_' + ts

        fname = os.path.join(self.out_dir, f"{stream_image_name}.npy")


        np.save(fname, frame_np)
        # print('saved stream image')


# --------------------------------------------------------------------------------------
# Main Widget (Controller for buttons and windows)
# --------------------------------------------------------------------------------------
class MyWidget(QtWidgets.QWidget):
    def __init__(self, pyrometer_app):
        super().__init__()
        self.setWindowTitle("RHEED Control")
        self.resize(600, 320)

        self.pyrometer_app = pyrometer_app # Will be passed as a function parameter to allow pyrometer control (i.e. accessing the temperature)

        # Buttons
        self.btn_start_live = QtWidgets.QPushButton("Start Live Feed")
        self.btn_stop_live = QtWidgets.QPushButton("Stop Live Feed")
        self.btn_acquire = QtWidgets.QPushButton("Acquire RHEED Image")
        self.btn_start_stream = QtWidgets.QPushButton("Start RHEED Stream")
        self.btn_stop_stream = QtWidgets.QPushButton("Stop RHEED Stream")
        self.btn_oscillation_settings = QtWidgets.QPushButton("Oscillation Settings")

        # Layout
        main_layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_start_live)
        row.addWidget(self.btn_stop_live)
        row.addWidget(self.btn_acquire)
        row.addWidget(self.btn_start_stream)
        row.addWidget(self.btn_stop_stream)
        row.addWidget(self.btn_oscillation_settings)
        main_layout.addLayout(row)

        # Live preview window
        self.live_window = LiveImageWindow(self.pyrometer_app, title="Live RHEED Feed")
        self.live_window.hide()

        # State
        self.live_worker: LiveViewWorker | None = None
        self.stream_worker: SaveStreamWorker | None = None
        self.selected_rect: QtCore.QRect | None = None

        # Connections
        self.btn_start_live.clicked.connect(self.start_live_feed)
        self.btn_stop_live.clicked.connect(self.stop_live_feed)
        self.btn_acquire.clicked.connect(self.acquire_single_image)
        self.btn_start_stream.clicked.connect(self.open_stream_dialog)
        self.btn_stop_stream.clicked.connect(self.stop_rheed_stream)
        self.btn_oscillation_settings.clicked.connect(self.open_oscillation_settings_dialog)

        # Initial UI state
        self._set_initial_ui()

       

    # ----------------- UI state helpers -----------------
    def _set_initial_ui(self):
        # Only "Start Live Feed" visible
        self.btn_start_live.setVisible(True)
        self.btn_oscillation_settings.setVisible(True)
        self.btn_stop_live.setVisible(False)
        self.btn_acquire.setVisible(False)
        self.btn_start_stream.setVisible(False)
        self.btn_stop_stream.setVisible(False)

    def _set_live_ui(self):
        # After starting live: show stop + acquire + start-stream
        self.btn_start_live.setVisible(False)
        self.btn_oscillation_settings.setVisible(False)
        self.btn_stop_live.setVisible(True)
        self.btn_acquire.setVisible(True)
        self.btn_start_stream.setVisible(True)
        self.btn_stop_stream.setVisible(False)

    def _set_stream_running_ui(self):
        # While RHEED stream is actively saving
        # Keep stop-live disabled to avoid confusion; user should stop the stream first
        self.btn_stop_live.setEnabled(False)
        self.btn_start_stream.setVisible(False)
        self.btn_stop_stream.setVisible(True)

    def _set_stream_stopped_ui(self):
        # Stream stopped; back to live UI
        self.btn_stop_live.setEnabled(True)
        self.btn_start_stream.setVisible(True)
        self.btn_stop_stream.setVisible(False)

    def closeEvent(self, e):
        
        super().closeEvent(e)

    
    # ----------------- Button handlers -----------------
    @QtCore.Slot()
    def start_live_feed(self):
        # If a stream is running, ignore (or you may stop it first)
        if self.stream_worker is not None:
            return
        # Kill any previous live worker (safety)
        if self.live_worker is not None:
            self.live_worker.stop()
            self.live_worker.wait()
            self.live_worker = None

        # Start live worker
        self.live_worker = LiveViewWorker(target_hz=1.0)
        self.live_worker.new_frame.connect(self.live_window.update_image)
        self.live_window.reset_intensity_plot()
        self.live_window.show()
        self.live_worker.start()
        print('starting live viewing stream')
        self._set_live_ui()

    @QtCore.Slot()
    def stop_live_feed(self):
        if self.stream_worker is not None:
            # Don't allow stopping live while a save-stream is running
            return
        if self.live_worker is not None:
            self.live_worker.stop()
            self.live_worker.wait()
            self.live_worker = None
        self.live_window.hide()
        self._set_initial_ui()
        print('stopped live viewing stream')

    @QtCore.Slot()
    def open_stream_dialog(self):
        dlg = StreamSettingsDialog(self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            duration, freq = dlg.get_values()
            if duration is None or freq is None or duration <= 0 or freq <= 0:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter positive numeric values.")
                return
            self.start_rheed_stream(duration, freq)

    @QtCore.Slot()
    def open_oscillation_settings_dialog(self):
        dlg = SelectRegionReferenceFileTemplate(self)
        dlg.regionSelected.connect(self._on_reference_region_selected)
        dlg.exec()

    @QtCore.Slot(QtCore.QRect, str)
    def _on_reference_region_selected(self, rect: QtCore.QRect, image_path: str):
        self.selected_rect = rect.normalized()
        self.live_window.set_intensity_rectangle(self.selected_rect)
        QtWidgets.QMessageBox.information(
            self,
            "Region set",
            f"Rect: x={rect.x()}, y={rect.y()}, w={rect.width()}, h={rect.height()}\nSource: {image_path}"
        )
    
    def _connect_preview(self, worker: _BaseCameraThread):
        # Ensure live window is visible and receives frames
        if not self.live_window.isVisible():
            self.live_window.show()
        worker.new_frame.connect(self.live_window.update_image)

    @QtCore.Slot()
    def start_rheed_stream(self, duration_s: float, freq_hz: float):
        # Stop live worker (the stream worker will also update the preview)
        if self.live_worker is not None:
            self.live_worker.stop()
            self.live_worker.wait()
            self.live_worker = None

        # Stop previous stream worker if any
        if self.stream_worker is not None:
            self.stream_worker.stop()
            self.stream_worker.wait()
            self.stream_worker = None

        # Start new stream worker
        # make a timestamped folder to save the stream images
        os.chdir(stream_images_folder)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_time_stream_images_folder = f"./{timestamp}"
        os.makedirs(current_time_stream_images_folder, exist_ok=True)

        self.stream_worker = SaveStreamWorker(self.pyrometer_app, duration_s, freq_hz, current_time_stream_images_folder)
        self.live_window.reset_intensity_plot()
        self._connect_preview(self.stream_worker)
        self.stream_worker.finished.connect(self._on_stream_finished)
        self.stream_worker.start()
        self._set_stream_running_ui()
        print('started acquisition stream')

    @QtCore.Slot()
    def stop_rheed_stream(self):
        if self.stream_worker is not None:
            self.stream_worker.stop()
            self.stream_worker.wait()
            self.stream_worker = None
        self._on_stream_finished()
        print('stopped acquisition stream')

    @QtCore.Slot()
    def _on_stream_finished(self):
        # When a recording stream finishes, return to live preview automatically
        if self.stream_worker is not None:
            self.stream_worker = None
        # Resume live preview
        if self.live_worker is None:
            self.live_worker = LiveViewWorker(target_hz=1.0)
            self.live_window.reset_intensity_plot()
            self._connect_preview(self.live_worker)
            self.live_worker.start()
        self._set_stream_stopped_ui()

    @QtCore.Slot()
    def acquire_single_image(self):
        # To avoid camera contention, temporarily stop live stream if active
        resume_live_after = False
        if self.stream_worker is not None:
            QtWidgets.QMessageBox.information(
                self, "Busy", "Cannot acquire a single image while RHEED stream is running. Stop the stream first."
            )
            return
        if self.live_worker is not None:
            resume_live_after = True
            self.live_worker.stop()
            self.live_worker.wait()
            self.live_worker = None

        saved = False
        try:
            
    
            # One-shot capture using vmbpy
            with VmbSystem.get_instance() as vmb:
                print('opening single image camera')
                cams = vmb.get_all_cameras()
                print(cams[0])
                if not cams:
                    raise RuntimeError("No camera detected.")
                cam = cams[0]
                with cam:
                    try:
                        cam.TriggerSource.set('Software')
                        cam.TriggerSelector.set('FrameStart')
                        cam.TriggerMode.set('On')
                        cam.AcquisitionMode.set('Continuous')
                    except Exception:
                        pass

                    frames = []
                    def _handler(c: Camera, s: Stream, f: Frame):
                        c.queue_frame(f)
                        frames.append(f)

                    cam.start_streaming(_handler)
                    try:
                        cam.TriggerSoftware.run()
                        print('acquired single image')
                        time.sleep(0.5)  # small wait for frame arrival
                    finally:
                        cam.stop_streaming()

                    if frames:
                        img = frames[-1].as_numpy_ndarray()
                        # print(img.dtype)
                        if img.dtype != np.uint8:
                            # Normalize to 8-bit if needed
                            a_min, a_max = float(img.min()), float(img.max())
                            if a_max <= a_min:
                                a_max = a_min + 1.0
                            img = ((img - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)
                        self._save_single_image(img)
                        
                        saved = True
                print('closing single image camera')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Acquire Failed", str(e))
            print('failed to acquire single image')

        # Resume live view if it was active
        if resume_live_after:
            self.start_live_feed()

        if saved:
            QtWidgets.QMessageBox.information(self, "Saved", "Single RHEED image saved.")

    def _save_single_image(self, frame_np: np.ndarray):
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

        temperature = get_pyrometer_temperature(self.pyrometer_app)
        # temperature = 'geo'
        # print(type(temperature))
        print(f'temperature = {temperature}')

        image_name = temperature + '_' + ts

        out = os.path.join(single_images_folder, f"{image_name}.npy")

        

        np.save(out, frame_np)
        print('saved single image')


def main():
    pyrometer_app = start_pyrometer()
    time.sleep(3)
    print('finished setting up pyrometer')
    app = QtWidgets.QApplication(sys.argv)
    w = MyWidget(pyrometer_app)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
