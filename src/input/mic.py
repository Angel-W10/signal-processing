import numpy as np
import pyaudio
import threading

# ─── Constants ────────────────────────────────────────────────────────────────
 
SAMPLE_RATE     = 44100   # Hz  — standard audio; Nyquist limit ~22 kHz
CHUNK_SIZE      = 1024    # samples per callback  (~23 ms of audio)
NUM_CHANNELS    = 1       # mono
BIT_DEPTH       = pyaudio.paInt16   # 16-bit integers from the ADC (−32768 … 32767)
INT16_MAX       = 32768.0           # used for normalisation to [-1.0, +1.0]

# ─── MicInput class ───────────────────────────────────────────────────────────
 
class MicInput:
    """
        Opens a PyAudio input stream in callback mode.
    
        Usage
        -----
            mic = MicInput()
            mic.start()
    
            # anywhere in your main loop:
            samples = mic.read()   # → np.ndarray, dtype=float32, shape=(CHUNK_SIZE,)
    
            mic.stop()
    
        Thread safety
        -------------
        The PyAudio callback fires on a private audio thread.
        We protect the shared buffer with a threading.Lock so the main thread
        can call read() at any time without a race condition.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_size:  int = CHUNK_SIZE,
        channels:    int = NUM_CHANNELS,
    ):
        self.sample_rate = sample_rate
        self.chunk_size  = chunk_size
        self.channels    = channels
 
        # PyAudio handle — one instance manages all audio I/O on this machine
        self._pa = pyaudio.PyAudio()
 
        # The stream object (created in start())
        self._stream = None
 
        # Latest chunk of normalised float32 samples.
        # Initialised to silence so read() is always safe to call.
        self._buffer: np.ndarray = np.zeros(chunk_size, dtype=np.float32)
 
        # Lock protecting _buffer between the audio thread and the main thread
        self._lock = threading.Lock()
 
        # Simple flag so we can query whether the stream is running
        self._running = False

# ── Private callback ──────────────────────────────────────────────────────
 
    def _callback(
        self,
        raw_bytes,   # the raw audio data from PyAudio — still plain bytes here
        frame_count, # how many frames (samples) are in this chunk
        time_info,   # dict with timing metadata (we ignore it)
        status_flags # any over/underflow flags from PortAudio
    ):
        """
            PyAudio calls this function automatically every ~23 ms on its own thread.
    
            Job:
            1. raw bytes  →  NumPy int16 array
            2. int16      →  float32 in [-1.0, +1.0]
            3. store in self._buffer under the lock
        """
        # Step 1 — interpret the raw bytes as a 1-D array of 16-bit integers.
        samples_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

        # Step 2 — normalise to float32.
        samples_float = samples_int16.astype(np.float32) / INT16_MAX

        # Step 3 — write to the shared buffer under the lock.
        # The lock ensures the main thread never reads a half-written buffer.
        with self._lock:
            self._buffer = samples_float

        # Returning paContinue tells PyAudio "keep the stream open".
        # Returning paComplete would close it.
        return (None, pyaudio.paContinue)

# ── Public interface ──────────────────────────────────────────────────────
 
    def start(self):
        """Open the audio stream and begin capturing."""

        self._stream = self._pa.open(
            format            = BIT_DEPTH,
            channels          = self.channels,
            rate              = self.sample_rate,
            input             = True,       # this is an INPUT stream (mic)
            frames_per_buffer = self.chunk_size,
            stream_callback   = self._callback,  # callback mode, not blocking
        )
        self._stream.start_stream()
        self._running = True
        print(f"[MicInput] stream open — {self.sample_rate} Hz, "
              f"chunk={self.chunk_size} samples "
              f"({1000 * self.chunk_size / self.sample_rate:.1f} ms)")
        
    def read(self) -> np.ndarray:
        """
            Return the most recent chunk as a float32 NumPy array.
    
            Shape:  (chunk_size,)
            Range:  [-1.0, +1.0]
            Note:   Returns the same chunk if called faster than the audio callback
                    fires — the caller is responsible for not over-reading.
        """
        with self._lock:
            # .copy() is important — without it the caller would hold a
            # reference to the internal buffer, which the audio thread
            # overwrites on the next callback.
            return self._buffer.copy()
        
    def stop(self):
        """
            Cleanly shut down the stream and release PortAudio resources.
        """
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
        self._running = False
        print("[MicInput] stream closed.")

    @property
    def running(self) -> bool:
        return self._running
    
# ── Context manager support ───────────────────────────────────────────────
    # Allows:  with MicInput() as mic:  ...
    # Automatically calls stop() on exit, even if an exception is raised.
 
    def __enter__(self):
        self.start()
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False   # don't suppress exceptions
    


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
# Run this file directly to verify mic capture is working:
#   python src/input/mic.py
 
if __name__ == "__main__":
    import time
 
    print("Smoke test — speak into your mic for 3 seconds...\n")
 
    with MicInput() as mic:
        for _ in range(6):          # 6 × 0.5 s = 3 s
            time.sleep(0.5)
            chunk = mic.read()
 
            # Basic statistics on the chunk — sanity-check the data
            peak    = np.max(np.abs(chunk))
            rms     = np.sqrt(np.mean(chunk ** 2))
            n_zeros = np.sum(chunk == 0)
 
            print(
                f"  samples={len(chunk)}  "
                f"dtype={chunk.dtype}  "
                f"peak={peak:.4f}  "
                f"rms={rms:.4f}  "
                f"zeros={n_zeros}"
            )
 
    print("\nDone. If peak > 0.0 when you spoke, mic capture is working ✓")
 
 