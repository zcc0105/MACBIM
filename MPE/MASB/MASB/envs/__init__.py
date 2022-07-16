from MASB.MASB.envs.simple_bidding import SimBid
import imp
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)
