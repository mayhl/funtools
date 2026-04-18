
import cmocean
import param

import holoviews as hv
from holoviews.plotting.util import process_cmap

import numpy as np

def colormap_dict():

    cmaps = _get_hv_cmaps()
    cmaps.update(_cmocean_cmaps())

    #for i in cmaps:
     #   for j in cmaps[i]:

    return cmaps 


def colormap_names():


    def get_all_cmaps(cmaps):
        tcmaps = [list(v.keys()) for v in cmaps.values()]
        n = np.sum([len(x) for x in tcmaps])
        tcmaps = np.concatenate(tcmaps)
        if not n == len(tcmaps): raise Exception('Non-unique name in provider %s.' % i)
        return tcmaps 

    cmaps = colormap_dict()
    all_cmaps = {k: get_all_cmaps(v) for k, v in cmaps.items()}
    all_cmaps = np.concatenate([["%s.%s" % (p, c) for c in v] for p, v in all_cmaps.items()])

    return all_cmaps


def _cmocean_cmaps(): 

    
    names = cmocean.cm.cmapnames
    
    div_maps = ['oxy', 'topo', 'balance', 'delta', 'curl', 'diff', 'tarn']
    
    for n in div_maps:
        if n not in names: raise Exception("Unknown cmocean color map %s." % n)

    def convert_to_hex(name, n=256):
        cmap = getattr(cmocean.cm, name)
        to_hex = lambda x:'#%02x%02x%02x' % tuple((x[0:3]*256).astype(int))
        s = np.linspace(0,1, n)
        return [to_hex(c) for c in cmap(s)]


    lambda : convert_to_hex(n) 

    cmaps = {'Standard' :{n: convert_to_hex(n) for n in names if not n in div_maps},
             'Diverging':{n: convert_to_hex(n) for n in div_maps}}
    
    return {"cmocean": cmaps}
    
def _get_hv_cmaps():

    cat_maps = {
        "Standard": "Uniform Sequential",
        "Diverging": "Diverging",
        "Categorical": "Categorical"
    }
    
    providers = ['colorcet', 'bokeh', 'matplotlib']


    _get_cmaps = lambda p, c: hv.plotting.util.list_cmaps(records=True,category=c, provider=p,reverse=False)
    _process_cmaps = lambda p, c: {cm.name: process_cmap(cm.name, provider=p) for cm in _get_cmaps(p, c)}

    cmaps = {p: {k: _process_cmaps(p, c) for k, c in cat_maps.items()} for p in providers}

    return cmaps




