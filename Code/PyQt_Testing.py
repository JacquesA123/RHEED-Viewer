import random
import sys
from PySide6 import QtCore, QtWidgets, QtGui
import time
from vmbpy import *
import numpy as np
import os
import datetime
from PyrometerControl import get_pyrometer_temperature

class StreamSettingsDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Stream Preferences")
        layout = QtWidgets.QFormLayout(self)
        
        self.duration_input = QtWidgets.QLineEdit(self)
        self.duration_input.setPlaceholderText("Enter stream duration in seconds")

        self.frequency_input = QtWidgets.QLineEdit(self)
        self.frequency_input.setPlaceholderText("Enter image frequency in Hz")
        
        layout.addRow("Stream Duration (seconds):", self.duration_input)
        layout.addRow("Image Frequency (Hz):", self.frequency_input)
        
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
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


class LiveImageWindow(QtWidgets.QWidget):
    # default title provided so callers don't have to always pass one
    def __init__(self, title="Live RHEED Feed"):
        super().__init__()
        self.setWindowTitle(title)   # use passed-in title (or default)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.image_label = QtWidgets.QLabel(self)
        self.layout.addWidget(self.image_label)
        
        self.setGeometry(100, 100, 800, 600)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)


    @QtCore.Slot(np.ndarray)
    def update_image(self, image_array):
        squeezed_image_array = image_array.squeeze()

        if squeezed_image_array.dtype != np.uint8:
            squeezed_image_array = squeezed_image_array.astype(np.uint8)

        height, width = squeezed_image_array.shape
        bytes_per_line = width

        qimage = QtGui.QImage(
            squeezed_image_array.copy().data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format_Grayscale8
        )

        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

        # resize the QLabel to fit the pixmap
        self.image_label.setFixedSize(width, height)

        # resize the whole window to fit the image (plus some margin for layout)
        self.resize(width, height)


class CameraWorker(QtCore.QThread):
    new_frame = QtCore.Signal(np.ndarray)

    def __init__(self, duration, frequency):
        super().__init__()
        self.duration = duration
        self.frequency = frequency
        self._running = True

    def run(self):
        with VmbSystem.get_instance() as vmb:
            cam = vmb.get_all_cameras()[0]
            print(f'Camera found at {cam}')
            with cam:
                cam.TriggerSource.set('Software')
                cam.TriggerSelector.set('FrameStart')
                cam.TriggerMode.set('On')
                cam.AcquisitionMode.set('Continuous')

                cam.start_streaming(self.handler)

                time_end = time.time() + self.duration
                while time.time() < time_end and self._running:
                    cam.TriggerSoftware.run()
                    time.sleep(1 / self.frequency)

                cam.stop_streaming()
                print('finished acquiring images')

    def stop(self):
        self._running = False

    def handler(self, cam: Camera, stream: Stream, frame: Frame):
        print('Frame acquired: {}'.format(frame), flush=True)
        cam.queue_frame(frame)
        image = frame.as_numpy_ndarray()
        now = datetime.datetime.now()
        valid_now = now.strftime('%Y-%m-%d_%H-%M-%S-%f.npy')

        temperature = get_pyrometer_temperature()
        print(temperature)
        os.chdir(stream_images_folder)
        np.save(valid_now, image)
        self.new_frame.emit(image)


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.AcquireImageButton = QtWidgets.QPushButton("Acquire RHEED Image")
        self.OpenRHEEDStreamPreferencesButton = QtWidgets.QPushButton("Start RHEED Stream")
        self.StopRHEEDStreamButton = QtWidgets.QPushButton("Stop RHEED Stream")
        


        # don't pre-create windows here; initialize to None
        self.live_window = None
        self.single_image_window = None
        self.worker = None


        self.StopRHEEDStreamButton.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.StopRHEEDStreamButton.hide()  # hide it initially

        

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.AcquireImageButton)
        self.layout.addWidget(self.OpenRHEEDStreamPreferencesButton)
        self.layout.addWidget(self.StopRHEEDStreamButton)

        self.AcquireImageButton.clicked.connect(self.acquire_RHEED_image)
        self.OpenRHEEDStreamPreferencesButton.clicked.connect(self.OpenRHEEDStreamPreferencesButton_clicked)
        self.StopRHEEDStreamButton.clicked.connect(self.stop_RHEED_camera_stream)

    @QtCore.Slot()
    def acquire_RHEED_image(self):
        with VmbSystem.get_instance() as vmb:
            cam = vmb.get_all_cameras()[0]
            print(f'Camera detected: {cam}')
            with cam:
                cam.TriggerSource.set('Software')
                cam.TriggerSelector.set('FrameStart')
                cam.TriggerMode.set('On')
                cam.AcquisitionMode.set('Continuous')

                try:
                    print('Preparing to acquire image')
                    frames = []
                    # append frames to list via lambda handler
                    cam.start_streaming(lambda c, s, f: frames.append(f))
                    cam.TriggerSoftware.run()
                    time.sleep(0.5)
                    # print(type(frames))
                    # print(len(frames))
                    # print(type(frames[0]))
                    # print(dir(frames[0]))
          
                    
                finally:
                    cam.stop_streaming()
                    print('finished acquiring image')

                if frames:
                    frame = frames[-1]   # take the last acquired frame
                    image = frame.as_numpy_ndarray()
                    now = datetime.datetime.now()
                    valid_now = now.strftime('%Y-%m-%d_%H-%M-%S-%f.npy')
                    temperature = get_pyrometer_temperature()
                    print(temperature)
                    os.chdir(single_images_folder)
                    np.save(valid_now, image)

                    # Open a window to show this single image
                    if self.single_image_window is None:
                        # pass a specific title for single images
                        self.single_image_window = LiveImageWindow(title="Single RHEED Image")
                    # update and show (show in case it was hidden)
                    self.single_image_window.update_image(image)
                    self.single_image_window.show()
                    # bring it to front
                    self.single_image_window.raise_()
                    self.single_image_window.activateWindow()

    @QtCore.Slot()
    def OpenRHEEDStreamPreferencesButton_clicked(self):
        dlg = StreamSettingsDialog()
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            duration, frequency = dlg.get_values()
            if duration is not None and frequency is not None:
                print(f"Stream Duration: {duration} seconds, Frequency: {frequency} Hz")
                
                self.run_RHEED_camera_stream(duration, frequency)

    @QtCore.Slot()
    @QtCore.Slot()
    def run_RHEED_camera_stream(self, duration, frequency):
        # Kill any previous worker
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        # If old window was closed, discard it
        if self.live_window is not None and not self.live_window.isVisible():
            self.live_window.deleteLater()
            self.live_window = None

        # Create a fresh live window
        if self.live_window is None:
            self.live_window = LiveImageWindow(title="Live RHEED Feed")
        self.live_window.show()

        # Start new worker
        self.worker = CameraWorker(duration, frequency)
        self.worker.new_frame.connect(self.live_window.update_image)
        self.worker.start()
    
    @QtCore.Slot()
    def stop_RHEED_camera_stream(self):
        # Stop the worker if running
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        # Close the live window if open
        if self.live_window:
            self.live_window.close()
            self.live_window = None

        print("RHEED stream stopped and window closed.")



# Configuration
single_images_folder = r"C:\Users\Lab10\Desktop\Automated RHEED Image Acquisition\Acquiring Images Via Python Script Tests\Single Images"
stream_images_folder = r"C:\Users\Lab10\Desktop\Automated RHEED Image Acquisition\Acquiring Images Via Python Script Tests\Stream Images"

if __name__ == "__main__":
    # os.chdir(image_folder)
    app = QtWidgets.QApplication([])
    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec())
# Functionb to start an experiment, where you define the duration and imaging frequency. At each imaging time, save image and temp and correlate them in a csv file.

# Live video feed at all times

# Get intensity of peaks and use that to plot RHEED oscillations