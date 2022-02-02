from csv import DictReader, DictWriter
from . import WindProcess


class CSVProcess(WindProcess):

    def __init__(self, file):
        super().__init__()
        assert file is not None, 'a data file must be provided'
        assert isinstance(file, str), 'file name must be a string'

        with open(file, 'r') as input_file:
            dict_reader = DictReader(input_file)
            data = [{k: float(v) for k, v in line.items()} for line in dict_reader]
        self._data = data

    def save(self, file):
        keys = self._data[0].keys() if len(self._data) > 0 else []
        with open(file, 'w') as output_file:
            dict_writer = DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self._data)

    def step(self):
        item = self._data[self._t]
        self._t = (self._t + 1) % len(self._data)
        return item
