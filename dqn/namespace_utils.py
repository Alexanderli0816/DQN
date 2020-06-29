import json


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def save_namespace(FLAGS, out_path):
    FLAGS_dict = vars(FLAGS)
    with open(out_path, 'w') as fp:
        # json.dump(FLAGS_dict, fp)
        json.dump(FLAGS_dict, fp, indent=4, sort_keys=True)


def load_namespace(in_path):
    with open(in_path, 'r') as fp:
        FLAGS_dict = json.load(fp)
    return Bunch(FLAGS_dict)


def enrich_options(options, TF_flags):
    for item in TF_flags.__flags.keys():
        if TF_flags.__flags[item].value is not None:
            options.__dict__[item] = TF_flags.__flags[item].value
    return options
