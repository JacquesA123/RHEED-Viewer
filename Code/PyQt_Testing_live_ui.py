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
import queue
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



# Default save locations (replace with your actual paths)
single_images_folder = os.path.expanduser(r"C:\Users\Lab10\Desktop\Automated RHEED Image Acquisition\RHEED Viewer\Acquiring Images Via Python Script Tests\Single Images")
stream_images_folder = os.path.expanduser(r"C:\Users\Lab10\Desktop\Automated RHEED Image Acquisition\RHEED Viewer\Acquiring Images Via Python Script Tests\Stream Images")
os.makedirs(single_images_folder, exist_ok=True)
os.makedirs(stream_images_folder, exist_ok=True)


# add near your imports
import queue

class PlotSaver(QtCore.QThread):
    """Runs Plotly/Kaleido image exports off the GUI thread, throttled."""
    def __init__(self, max_fps=1, parent=None):
        super().__init__(parent)
        self.q = queue.Queue(maxsize=2)   # small buffer to apply backpressure
        self._running = True
        self._min_interval = 1.0 / max(0.1, float(max_fps))
        self._last = 0.0

    def submit(self, x, y, path):
        try:
            self.q.put_nowait((np.asarray(x), np.asarray(y), path))
        except queue.Full:
            pass  # drop if we’re behind

    def stop(self):
        self._running = False
        try: self.q.put_nowait(None)
        except queue.Full: pass
        self.wait()

    def run(self):
        import time
        import plotly.express as px
        while self._running:
            item = self.q.get()
            if item is None:
                break
            x, y, path = item
            now = time.time()
            if now - self._last < self._min_interval:
                continue
            self._last = now
            try:
                fig = px.line(x=x, y=y, labels={'x':'x', 'y':'sin(x)'}, title="Sine Wave")
                fig.write_image(path, width=800, height=500, scale=1)
            except Exception as e:
                print("plot save error:", e)


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
    def __init__(self, pyrometer_app, title="Live RHEED Feed", parent=None):
        super().__init__(parent)
        self.pyrometer_app = pyrometer_app

        self.setWindowTitle(title)
        self.resize(656, 492)  # use resize instead of setGeometry for a top-level

        # --- Layout: create it WITHOUT a parent, then set once ---
        layout = QtWidgets.QVBoxLayout()
        # layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(8)

        # --- Image area ---
        self.image_label = QtWidgets.QLabel()
        self.test_image = QtWidgets.QLabel()
        # self.test_image.setPixmap(QtGui.QPixmap(r"C:\Users\Lab10\Pictures\Screenshots\Screenshot (1).png"))
        layout.addWidget(self.image_label)
        layout.addWidget(self.test_image)




        self.setLayout(layout)
       
        







    @QtCore.Slot(np.ndarray)
    def update_image(self, image_array: np.ndarray):

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
        

        self.image_label.setPixmap(pix) # Update the QLabel widget with the new pixmap (image)
        self.image_label.resize(pix.size()) # Resize the label so the window fits the pixmap size exactly

        # Show the plot
        plot_path = r"C:\Users\Lab10\Desktop\Automated RHEED Image Acquisition\RHEED Viewer\Acquiring Images Via Python Script Tests\Stream Images\sine_2025-09-24_09-12-37-632123.png"
        plot_pixmap = QtGui.QPixmap(plot_path)
        self.test_image.setPixmap(plot_pixmap)
        self.image_label.resize(plot_pixmap.size())

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
                                print('Trigger fired', flush=True)
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
        print(type(temperature))
        print(temperature)

        stream_image_name = temperature + '_' + ts

        fname = os.path.join(self.out_dir, f"{stream_image_name}.npy")


        np.save(fname, frame_np)
        print('saved stream image')


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

        # Layout
        main_layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_start_live)
        row.addWidget(self.btn_stop_live)
        row.addWidget(self.btn_acquire)
        row.addWidget(self.btn_start_stream)
        row.addWidget(self.btn_stop_stream)
        main_layout.addLayout(row)

        # Live preview window
        self.live_window = LiveImageWindow(self.pyrometer_app, title="Live RHEED Feed")
        self.live_window.hide()

        # State
        self.live_worker: LiveViewWorker | None = None
        self.stream_worker: SaveStreamWorker | None = None

        # Connections
        self.btn_start_live.clicked.connect(self.start_live_feed)
        self.btn_stop_live.clicked.connect(self.stop_live_feed)
        self.btn_acquire.clicked.connect(self.acquire_single_image)
        self.btn_start_stream.clicked.connect(self.open_stream_dialog)
        self.btn_stop_stream.clicked.connect(self.stop_rheed_stream)

        # Initial UI state
        self._set_initial_ui()

        # start saver (e.g., 1 PNG per second)
        self.plot_saver = PlotSaver(max_fps=1, parent=self)
        self.plot_saver.start()

        # snapshot timer (don’t do it every frame!)
        self.snapshot_timer = QtCore.QTimer(self)
        self.snapshot_timer.setInterval(1000)  # ms
        self.snapshot_timer.timeout.connect(self._snapshot_plot)
        self.snapshot_timer.start()

    # ----------------- UI state helpers -----------------
    def _set_initial_ui(self):
        # Only "Start Live Feed" visible
        self.btn_start_live.setVisible(True)
        self.btn_stop_live.setVisible(False)
        self.btn_acquire.setVisible(False)
        self.btn_start_stream.setVisible(False)
        self.btn_stop_stream.setVisible(False)

    def _set_live_ui(self):
        # After starting live: show stop + acquire + start-stream
        self.btn_start_live.setVisible(False)
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
        # cleanly stop worker
        if hasattr(self, "plot_saver") and self.plot_saver is not None:
            self.plot_saver.stop()
        super().closeEvent(e)

    @QtCore.Slot()
    def _snapshot_plot(self):
        # build your data quickly on GUI thread
        x = np.linspace(0, 2*np.pi, 500)
        y = np.sin(x)
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        out = os.path.join(stream_images_folder, f"sine_{ts}.png")
        # hand off heavy work to the saver thread
        self.plot_saver.submit(x, y, out)

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
        print(type(temperature))
        print(temperature)

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
