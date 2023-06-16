import json


class Feature(object):
    def __init__(self, name, sql_file,
                 temporal=True, raw_sql=None):
        self.name = name
        self.sql_file = sql_file
        self.temporal = temporal

        if not raw_sql:
            with open(sql_file, 'r') as f:
                raw_sql = f.read()

        self.raw_sql = raw_sql

class FeatureSet(object):
    def __init__(self, feature_list):
        self.features = feature_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def serialize(self, output_path):
        features_dict = {}
        for f in self.features:
            fdict = vars(f)
            features_dict[fdict['name']] = fdict

        with open(output_path, 'w') as fh:
            json.dump(features_dict, fh)

    @classmethod
    def from_file(cls, filepath):
        feature_list = []
        with open(filepath, 'r') as fh:
            d = json.load(fh)
        for name, fd in d.items():
            name = fd.pop("name")
            sql_file = fd.pop("sql_file")
            f = Feature(name, sql_file, **fd)
            feature_list.append(f)
        return cls(feature_list)
