from csv import DictReader, DictWriter
from . import WindProcess


class CSVProcess(WindProcess):
    """
    CSVProcess is a wind process that reads data from a `.csv`-file. The file must have the following format:

    First line contains the names of the atmospheric measurements, e.g., wind_speed or wind_direction
    The following lines contain the corresponding measurements. Each time `step` is called, the data from the next
    unused line is returned
    """

    def __init__(self, file):
        super().__init__()
        assert file is not None, 'a data file must be provided'
        assert isinstance(file, str), 'file name must be a string'

        with open(file, 'r') as input_file:
            dict_reader = DictReader(input_file)
            data = [{k: float(v) for k, v in line.items()} for line in dict_reader]
        self._data = data
        self._t = 0

    def save(self, file):
        keys = self._data[0].keys() if len(self._data) > 0 else []
        with open(file, 'w') as output_file:
            dict_writer = DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self._data)

    def step(self):
        item = self._data[self._t]
        self._t = (self._t + 1) % len(self._data)  # if there are no more lines in the data, start from the beginning
        return item

    def reset(self):
        self._t = 0
