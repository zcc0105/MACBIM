import numpy as np
import imp
import os.path as osp


def load(name):
    name = 'envs/' + name
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)


class BaseScenario(object):
    def make_world(self):
        raise NotImplementedError()

    def reset_world(self, world):
        raise NotImplementedError()
