import time, datetime
import json


class Timer:
    def __init__(self):
        self.last = time.time()

    def spot(self):
        now = time.time()
        period = now - self.last
        self.last = now
        return str(datetime.timedelta(seconds=int(period)))


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def save_namespace(FLAGS, out_path):
    FLAGS_dict = vars(FLAGS)
    with open(out_path, 'w') as fp:
        json.dump(FLAGS_dict, fp)


def load_namespace(in_path):
    with open(in_path, 'r') as fp:
        FLAGS_dict = json.load(fp)
    return Bunch(FLAGS_dict)


def metric2str(m):
    return ['%s: %.2f' % (k, v * 100) for k, v in
            sorted(m.items()) if k.endswith('hits')]

