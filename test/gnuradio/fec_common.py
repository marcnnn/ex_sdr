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


# -- flowgraph helpers -----------------------------------------------------------------

def run_chain(blks, data, src="b", snk="b", extra_inputs=None):
    """Runs one stream chain over `data` (a list) with vector source / sink
    of types src, snk ("b", "f"); returns the sink data."""
    from gnuradio import blocks, gr

    srcs = {"b": blocks.vector_source_b, "f": blocks.vector_source_f}
    snks = {"b": blocks.vector_sink_b, "f": blocks.vector_sink_f}
    s = srcs[src](list(data), False)
    k = snks[snk]()
    tb = gr.top_block()
    tb.connect(s, *blks, k)
    tb.run()
    return list(k.data())


def run_tagged(blk, packets, key="packet_len", src="b", snk="b"):
    """Packets (lists) through a tagged stream block; returns the output packets."""
    import pmt
    from gnuradio import blocks, gr

    def tag(offset, value):
        t = gr.tag_t()
        t.offset = offset
        t.key = pmt.intern(key)
        t.value = pmt.from_long(value)
        return t

    data, tags, off = [], [], 0
    for p in packets:
        tags.append(tag(off, len(p)))
        data += list(p)
        off += len(p)
    srcs = {"b": blocks.vector_source_b, "f": blocks.vector_source_f}
    snks = {"b": blocks.vector_sink_b, "f": blocks.vector_sink_f}
    s = srcs[src](data, False, 1, tags)
    k = snks[snk]()
    tb = gr.top_block()
    tb.connect(s, blk, k)
    tb.run()
    out = list(k.data())
    lens = [pmt.to_python(t.value) for t in k.tags() if pmt.symbol_to_string(t.key) == key]
    res, pos = [], 0
    for n in lens:
        res.append(out[pos : pos + n])
        pos += n
    return res


def run_pdu(blk, pdus, make_vec, out_ports=("out",), wait=0.5):
    """PDUs (lists of items) through a message block; `make_vec(list)` builds
    the PMT vector. Returns {port: [list of items]}."""
    import time

    import pmt
    from gnuradio import blocks, gr

    dbgs = {}
    tb = gr.top_block()
    for p in out_ports:
        d = blocks.message_debug()
        dbgs[p] = d
        tb.msg_connect(blk, p, d, "store")
    tb.start()
    for data in pdus:
        blk.to_basic_block()._post(pmt.intern("in"), pmt.cons(pmt.make_dict(), make_vec(list(data))))
    time.sleep(wait)
    tb.stop()
    tb.wait()
    out = {}
    for p, d in dbgs.items():
        out[p] = []
        for i in range(d.num_messages()):
            v = pmt.cdr(d.get_message(i))
            out[p].append(
                list(pmt.u8vector_elements(v)) if pmt.is_u8vector(v) else list(pmt.f32vector_elements(v))
            )
    return out
