"""Shared helpers of the fec golden scripts: calling gr-fec's generic
encoder / decoder objects frame by frame from Python (their generic_work
takes raw buffers, wrapped here in capsules)."""
import ctypes

import numpy as np

_new = ctypes.pythonapi.PyCapsule_New
_new.restype = ctypes.py_object
_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def cap(a):
    return _new(a.ctypes.data, None, None)


def work(obj, inp, out):
    """One generic_work call; inp, out are contiguous numpy arrays."""
    obj.generic_work(cap(inp), cap(out))
    return out


def bitstr(bits):
    return "".join(str(int(b) & 1) for b in bits)


def hexs(a):
    return bytes(np.asarray(a, dtype=np.uint8).tolist()).hex()


def encode_frames(enc, frames_in):
    """Runs the encoder over successive frames; returns the outputs."""
    outs = []
    for fr in frames_in:
        inp = np.ascontiguousarray(fr, dtype=np.uint8)
        out = np.zeros(enc.get_output_size(), dtype=np.uint8)
        outs.append(work(enc, inp, out).copy())
    return outs
