from __future__ import annotations

import ctypes
import getpass
import os


def set_process_title(name: str | None = None) -> str:
    """Set a short process title for GPU process monitors such as nvidia-smi."""
    title = name or os.environ.get("MHW_PROCESS_NAME") or getpass.getuser()

    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
        return title
    except Exception:
        pass

    if os.name == "posix":
        try:
            libc = ctypes.CDLL(None)
            pr_set_name = 15
            encoded = title.encode("utf-8")[:15]
            libc.prctl(pr_set_name, ctypes.c_char_p(encoded), 0, 0, 0)
        except Exception:
            pass

    return title
