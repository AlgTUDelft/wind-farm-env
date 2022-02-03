import floris.tools as ft
import numpy as np
from gym.envs.classic_control import rendering
from .wind_map import WindMap
from typing import Union, Tuple


class FarmVisualization:
    """
    FarmVisualization handles rendering of a wind farm.
    """

    def __init__(self, fi: ft.floris_interface.FlorisInterface,
                 resolution: Union[int, Tuple[int, int]] = 64, viewer_width=640, dpi=80,
                 x_bounds=None, y_bounds=None,
                 margins=None, units='diam', color_map=None):
        """
        Initializes a wind farm visualization

        :param fi: FlorisInterface to render
        :param resolution: rendering is done in blocks, this is the number of blocks along each axis; if a single value
        is given, it will be used for the larger  dimension of the  wind farm, and the other axis will have a resolution
        to keep the blocks as close to  squares as possible
        :param viewer_width: view port width in pixels
        :param dpi: DPI for rendering
        :param x_bounds: coordinate bounds along the x axis; if None, will be derived automatically
        :param y_bounds: coordinate bounds along the y axis; if None, will be derived automatically
        :param margins: margins to add to automatically derived bounds; defaults to two turbine diameters on each side
            except east,  where it is ten diameters; this is done so that the wakes can still be seen in the east
        :param units: units of measurement for the margins; 'diam' means turbine diameters and 'm' --- meters
        :param color_map: matplotlib color map for rendering
        """
        self._floris_interface = fi
        farm = self._floris_interface.floris.farm
        turbine_coordinates = farm.flow_field.turbine_map.coords
        if x_bounds and y_bounds:
            self.x_bounds = x_bounds
            self.y_bounds = y_bounds
        else:
            if margins:
                m = np.array(margins)
            else:
                m = np.array((2, 10, 2, 2))
            if units == 'diam':
                m = m * farm.flow_field.max_diameter
            elif units == 'm':
                pass
            else:
                raise NotImplementedError
            if x_bounds is None:
                x = [turbine.x1 for turbine in turbine_coordinates]
                self.x_bounds = (min(x) - m[3], max(x) + m[1])
            if y_bounds is None:
                y = [turbine.x2 for turbine in turbine_coordinates]
                self.y_bounds = (min(y) - m[2], max(y) + m[0])
        if hasattr(resolution, '__len__'):
            self.x_resolution = resolution[0]
            self.y_resolution = resolution[1]
        else:
            w = self.width
            h = self.height
            if w >= h:
                self.x_resolution = resolution
                self.y_resolution = int(resolution / w * h)
            else:
                self.y_resolution = resolution
                self.x_resolution = int(resolution / h * w)
        self.dpi = dpi
        self.viewer_width = viewer_width
        self.viewer_height = int(self.viewer_width / (self.width / self.height))
        self.plot_width = self.viewer_width / self.dpi
        self.plot_height = self.plot_width / (self.width / self.height)
        self.viewer = rendering.Viewer(self.viewer_width, self.viewer_height)
        self.color_map = color_map
        self._hub_height = self._floris_interface.floris.farm.flow_field.turbine_map.turbines[0].hub_height

        # add the wind map
        self.wind_map = None

    @property
    def width(self):
        return self.x_bounds[1] - self.x_bounds[0]

    @property
    def height(self):
        return self.y_bounds[1] - self.y_bounds[0]

    def render(self, return_rgb_array=False):
        self._floris_interface.reinitialize_flow_field()
        self._floris_interface.calculate_wake()
        cut_plane = self._floris_interface.get_hor_plane(x_resolution=self.x_resolution,
                                                         y_resolution=self.y_resolution,
                                                         x_bounds=self.x_bounds,
                                                         y_bounds=self.y_bounds,
                                                         height=self._hub_height)
        wind_direction = self._floris_interface.floris.farm.wind_map.input_direction[0]
        turbines_raw_data = [
            (np.deg2rad(turbine.yaw_angle - wind_direction - 90), coordinates, turbine.rotor_radius)
            for coordinates, turbine
            in self._floris_interface.floris.farm.flow_field.turbine_map.items
        ]
        if self.wind_map is None:
            self.wind_map = WindMap((self.plot_width, self.plot_height), self.dpi, cut_plane, turbines_raw_data,
                                    self.color_map)
            self.wind_map.scale_x = self.viewer_width / self.wind_map.width
            self.wind_map.scale_y = self.viewer_height / self.wind_map.height
            self.viewer.add_geom(self.wind_map)
        else:
            self.wind_map.update_image(cut_plane, turbines_raw_data)

        return self.viewer.render(return_rgb_array)

    def close(self):
        self.wind_map.close()
        self.viewer.close()
