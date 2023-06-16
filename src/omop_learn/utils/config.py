import json
from types import SimpleNamespace


class Config(SimpleNamespace):
    def __init__(self, d):
        super().__init__(**d)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as fh:
            d = json.load(fh)
        return cls(d)

    def serialize(self, output_path):
        with open(output_path, 'w') as fh:
            json.dump(vars(self), fh)
