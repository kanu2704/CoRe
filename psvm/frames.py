"""Image storage for CoRe's image buffers.

CoRe keeps every 300x300 RGB frame in plain numpy arrays (~73 GB per MetaWorld
run). FrameArray keeps the same frames losslessly, so every pixel the reward
model or the VLM sees is identical to the original code:

  'ram'  plain numpy array (original CoRe behaviour)
  'png'  each frame PNG-encoded in memory (~2.5x smaller on MetaWorld frames)
  'disk' numpy memmap file (frames live on disk instead of RAM)
"""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

_DECODE_POOL = ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2))))


def encode_png(frame, level=1):
    ok, buf = cv2.imencode('.png', np.ascontiguousarray(frame), [cv2.IMWRITE_PNG_COMPRESSION, int(level)])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def decode_png(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


class Frame:
    """A rendered frame that can be stored in several FrameArrays while being
    PNG-encoded at most once."""
    __slots__ = ('raw', '_png', '_level')

    def __init__(self, raw, level=1):
        self.raw = raw
        self._png = None
        self._level = level

    def png(self):
        if self._png is None:
            self._png = encode_png(self.raw, self._level)
        return self._png


class FrameArray:
    """Fixed number of HxWx3 uint8 frames addressed by a flat integer index."""

    def __init__(self, n, height, width, backend='ram', png_level=1, disk_dir=None, name='frames'):
        self.n = int(n)
        self.shape = (int(height), int(width), 3)
        self.backend = backend
        self.png_level = png_level
        self._path = None
        if backend == 'ram':
            self._data = np.empty((self.n, *self.shape), dtype=np.uint8)
        elif backend == 'png':
            self._data = [None] * self.n
        elif backend == 'disk':
            disk_dir = disk_dir or os.environ.get('PSVM_DISK_DIR', '/tmp/psvm_frames')
            os.makedirs(disk_dir, exist_ok=True)
            self._path = os.path.join(disk_dir, f'{name}_{os.getpid()}_{id(self)}.u8')
            self._data = np.memmap(self._path, dtype=np.uint8, mode='w+', shape=(self.n, *self.shape))
        else:
            raise ValueError(f"unknown frame backend: {backend}")

    def put(self, idx, frame):
        if self.backend == 'png':
            if isinstance(frame, Frame):
                self._data[idx] = frame.png()
            else:
                self._data[idx] = encode_png(frame, self.png_level)
        else:
            raw = frame.raw if isinstance(frame, Frame) else frame
            self._data[idx] = raw

    def put_many(self, idxs, frames):
        idxs = np.asarray(idxs).reshape(-1)
        frames = np.asarray(frames).reshape(-1, *self.shape)
        if self.backend == 'png':
            for i, f in zip(idxs, frames):
                self._data[int(i)] = encode_png(f, self.png_level)
        else:
            self._data[idxs] = frames

    def take(self, idxs):
        """Return frames at `idxs` (any integer array) with shape idxs.shape + (H, W, 3)."""
        idxs = np.asarray(idxs)
        if self.backend != 'png':
            return np.asarray(self._data[idxs])
        flat = idxs.reshape(-1)
        out = np.empty((len(flat), *self.shape), dtype=np.uint8)
        blobs = [self._data[int(i)] for i in flat]
        if any(b is None for b in blobs):
            raise IndexError("FrameArray: reading a frame that was never stored")

        def _dec(j):
            out[j] = decode_png(blobs[j])

        if len(flat) >= 64:
            list(_DECODE_POOL.map(_dec, range(len(flat))))
        else:
            for j in range(len(flat)):
                _dec(j)
        return out.reshape(*idxs.shape, *self.shape)

    def nbytes(self):
        if self.backend == 'png':
            return sum(len(b) for b in self._data if b is not None)
        return self._data.nbytes

    def release(self):
        """Free all stored frames."""
        if self.backend == 'png':
            self._data = [None] * self.n
        else:
            self._data = None
            if self._path and os.path.exists(self._path):
                os.remove(self._path)
