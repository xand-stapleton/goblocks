import ctypes
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

goblocks_root = Path(__file__).resolve().parents[2]

libpath = goblocks_root / Path("server/src/lib/librecursive.so")

lib = ctypes.cdll.LoadLibrary(str(libpath))

CreateRD = lib.CreateRD
CreateRD.argtypes = [ctypes.c_char_p]
CreateRD.restype = ctypes.c_longlong

FreeRD = lib.FreeRD
FreeRD.argtypes = [ctypes.c_longlong]
FreeRD.restype = None

RunRequest = lib.RunRequest
RunRequest.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong)]
RunRequest.restype = ctypes.POINTER(ctypes.c_double)

FreeResult = lib.FreeResult
FreeResult.argtypes = [ctypes.POINTER(ctypes.c_double)]
FreeResult.restype = None


class BlockHandle:
    """
    This class manages:
      - Handle creation (via CreateRD)
      - Launching requests (via RunRequest)
      - Freeing the handle (via FreeRD)

    Parameters correspond to the JSON configuration used by the C library.
    """

    def __init__(
        self,
        k1_max: int = 10,
        k2_max: int = 10,
        ell_min: int = 0,
        ell_max: int = 6,
        d: int = 3,
        nmax: int = 8,
        cache_dir: str = "cache",
        use_precomputed_phi1: bool = True,
    ) -> None:
        self.handle_params: Dict[str, Any] = {
            "k1_max": k1_max,
            "k2_max": k2_max,
            "ell_min": ell_min,
            "ell_max": ell_max,
            "d": d,
            "nmax": nmax,
            "cache_dir": cache_dir,
            "use_precomputed_phi1": use_precomputed_phi1,
        }

        # Create C handle
        encoded = json.dumps(self.handle_params).encode("utf-8")
        self.handle: int = CreateRD(ctypes.c_char_p(encoded))

    def run_request(
        self,
        command: str = "recurse_and_evaluate_df",
        deltas: List[float] = [5.1],
        ells: List[int] = [3],
        block_types: List[str] = ["+", "-"],
        delta_12: float = 1.6,
        delta_34: float = 1.2,
        delta_ave_23: float = 3.1,
        max_iterations: int = 100,
        tol: float = 1e-4,
        r: float = 0.1715728753,
        eta: float = 1.0,
        nmax: int = 8,
        normalise: bool = True,
        use_numerical_derivs: bool = False,
    ) -> Optional[List[float]]:
        """
        Execute a computation request on the underlying C handle.

        Returns
        -------
        list[float] | None
            Result array from the C library, or None on error.
        """

        request: Dict[str, Any] = {
            "command": command,
            "handle": self.handle,
            "deltas": deltas,
            "ells": ells,
            "block_types": block_types,
            "delta_12": delta_12,
            "delta_34": delta_34,
            "delta_ave_23": delta_ave_23,
            "max_iterations": max_iterations,
            "tol": tol,
            "r": r,
            "eta": eta,
            "nmax": nmax,
            "normalise": normalise,
            "use_numeric_derivs": use_numerical_derivs,
        }

        # Warn inline if request parameters will be overridden by handle_params
        overridden = []
        for key in request.keys() & self.handle_params.keys():
            rv = request[key]
            hv = self.handle_params[key]
            if rv != hv:
                overridden.append((key, rv, hv))

        if overridden:
            print("[WARNING] The following parameters passed to run_request() "
                  "are overridden by handle_params:")
            for key, rv, hv in overridden:
                print(f"  - {key}: requested={rv!r} → using handle={hv!r}")

        request.update(self.handle_params)
        return self._run_request(request)

    @staticmethod
    def _run_request(req_obj: Dict[str, Any]) -> Optional[List[float]]:
        """Internal: JSON-encode request -> call C -> convert result"""
        payload = json.dumps(req_obj).encode("utf-8")

        out_len = ctypes.c_longlong()
        ptr = RunRequest(ctypes.c_char_p(payload), ctypes.byref(out_len))

        if not ptr:
            print("RunRequest returned NULL or error.")
            return None

        count = out_len.value
        result = [ptr[i] for i in range(count)]

        # Free memory allocated by the C library
        FreeResult(ptr)
        return result

    def free_handle(self) -> None:
        """Free the underlying C handle."""
        FreeRD(self.handle)



class BlockHandlePool:
    """
    Thread-safe pool of BlockHandle objects.

    Features
    --------
    - Reuses idle handles
    - Allocates new handles when all are in use
    - Safe for concurrent / threaded workloads
    - Provides a context manager for easy usage

    Parameters
    ----------
    handle_params : dict, optional
        Passed directly to new BlockHandle() instances.
    """

    def __init__(self, handle_params: Optional[Dict[str, Any]] = None) -> None:
        self.lock = threading.Lock()
        self.pool: List[List[Any]] = []  # [BlockHandle, busy_flag]
        self.handle_params = handle_params or {}

    def _create_new_handle(self) -> BlockHandle:
        """Create a new handle and mark it busy immediately."""
        handle = BlockHandle(**self.handle_params)
        self.pool.append([handle, True])
        return handle

    def acquire_handle(self) -> BlockHandle:
        """
        Acquire a handle from the pool.

        Returns
        -------
        BlockHandle
            A usable handle instance.
        """
        with self.lock:
            for entry in self.pool:
                handle, busy = entry
                if not busy:
                    entry[1] = True
                    return handle

            # no free handles -> allocate a new one
            return self._create_new_handle()

    def release_handle(self, handle: BlockHandle) -> None:
        """
        Mark a handle as free.

        Raises
        ------
        RuntimeError
            If the handle is not tracked by this pool.
        """
        with self.lock:
            for entry in self.pool:
                if entry[0] is handle:
                    entry[1] = False
                    return

            raise RuntimeError("Attempted to release a handle not in this pool.")

    def close_all(self) -> None:
        """Free all C handles and empty the pool."""
        with self.lock:
            for handle, _ in self.pool:
                handle.free_handle()
            self.pool.clear()

    @contextmanager
    def get(self) -> Generator[BlockHandle, None, None]:
        """
        Context manager for automatic acquire/release.

        Example
        -------
        >>> with pool.get() as handle:
        ...     result = handle.run_request()
        """
        h = self.acquire_handle()
        try:
            yield h
        finally:
            self.release_handle(h)


if __name__ == "__main__":
    pool = BlockHandlePool()

    # Using context manager (recommended)
    with pool.get() as h:
        print("Result:", h.run_request())

    # Manual acquire/release
    # h = pool.acquire_handle()
    # print("Manual result:", h.run_request())
    # pool.release_handle(h)

    # Free all C handles
    pool.close_all()
