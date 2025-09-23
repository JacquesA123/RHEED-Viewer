

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import sys
import io
import plotly.express as px
from PySide6 import QtWidgets, QtGui

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

def obtain_pixmap():
    fig = Figure(figsize=(3, 2), dpi=100)
    print('initialized fig')
    ax = fig.add_subplot(111)
    x = np.linspace(0, 2*np.pi, 100)
    ax.plot(x, np.sin(x))
    ax.set_title("Sine")
    pixmap = figure_to_pixmap(fig)
    return pixmap

def plotly_figure_to_qpixmap(fig, width=600, height=400) -> QtGui.QPixmap:
    """Render a Plotly figure to QPixmap without writing to disk."""
    # Export figure to PNG bytes (requires kaleido)
    png_bytes = fig.to_image(format="png", width=width, height=height)
    # Wrap bytes in a QPixmap
    pixmap = QtGui.QPixmap()
    pixmap.loadFromData(png_bytes, "PNG")
    return pixmap