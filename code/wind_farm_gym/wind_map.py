import floris.tools as ft

import pyglet
from pyglet.sprite import Sprite

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg


class WindMap(Sprite):

    def __init__(self, size, dpi, cut_plane, turbines_raw_data, wind_speed_limits=None, color_map='GnBu_r'):
        self.fig = plt.figure(figsize=size, dpi=dpi)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.ax = self.fig.gca()
        self.ax.set_axis_off()
        self.ax.xaxis.set_major_locator(plt.NullLocator())
        self.ax.yaxis.set_major_locator(plt.NullLocator())
        self.canvas = FigureCanvasAgg(self.fig)
        self.color_map = color_map
        if wind_speed_limits is None:
            self.min_wind_speed = self.max_wind_speed = None
        elif hasattr(wind_speed_limits, '__len__') and len(wind_speed_limits) > 1:
            self.min_wind_speed = wind_speed_limits[0]
            self.max_wind_speed = wind_speed_limits[1]
        else:
            raise NotImplementedError
        img = self.find_image(cut_plane, turbines_raw_data)
        super().__init__(img)

    def find_image(self, cut_plane, turbines_raw_data=None):
        self.clear()
        ft.visualization.visualize_cut_plane(cut_plane, self.ax, self.min_wind_speed, self.max_wind_speed,
                                             self.color_map)
        lines = []
        for angle, coordinates, radius in turbines_raw_data:
            line = np.array([[0, -radius], [0, radius]])
            small_line = np.array([[radius / 5, -radius / 5], [radius / 5, radius / 5]])
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, s], [-s, c]])
            line = line @ rotation
            small_line = small_line @ rotation
            shift = np.array([[coordinates.x1, coordinates.x2], [coordinates.x1, coordinates.x2]])
            line = line + shift
            small_line = small_line + shift
            lines.append(line)
            lines.append(small_line)
        lc = LineCollection(lines, color="black", lw=4)
        self.ax.add_collection(lc)
        raw_data, size = self.canvas.print_to_buffer()
        width = size[0]
        height = size[1]
        return pyglet.image.ImageData(width, height, 'RGBA', raw_data, -4 * width)

    def update_image(self, cut_plane, turbines_raw_data=None):
        self.image = self.find_image(cut_plane, turbines_raw_data)

    def render(self):
        self.draw()

    def clear(self):
        self.ax.collections = []

    def close(self):
        plt.close(self.fig)
