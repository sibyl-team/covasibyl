import zstandard as zstd
import dill

def load_dill_zstd(fname, decompr=None):
    if decompr is None:
        decompr=zstd.ZstdDecompressor()
    with open(fname,"rb") as f:
        with decompr.stream_reader(f) as strm:
            dat=dill.load(strm)
    return dat

def save_dill_zstd(mobj, fname, compr=None, level=5):
    if compr is None:
        compr=zstd.ZstdCompressor(level=level)
    with open(fname,"wb") as f:
        with compr.stream_writer(f) as sw:
            dill.dump(mobj, sw)