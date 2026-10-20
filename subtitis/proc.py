"""Proces po PID-u, bez pretrage po imenu (krug 29).

PID se na Windows-u ponovo dodeljuje: zaostao `tmp/server.pid` od pale
instance ume da pokazuje na tuđ proces koji je u međuvremenu dobio isti broj.
Zato se uz PID uvek pamti i VREME POKRETANJA procesa, i proces se smatra istim
samo ako se oba poklapaju.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259


def start_time(pid: int) -> int | None:
    """Vreme pokretanja procesa (FILETIME, jedinice od 100 ns), ili None ako
    proces ne postoji ili mu se ne može pristupiti."""
    if sys.platform != "win32":  # pragma: no cover — alat radi na Windows-u
        try:
            return int(os.stat(f"/proc/{int(pid)}").st_ctime_ns // 100)
        except OSError:
            return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return None
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != STILL_ACTIVE:
            return None
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not k32.GetProcessTimes(h, ctypes.byref(created), ctypes.byref(exited),
                                   ctypes.byref(kernel), ctypes.byref(user)):
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        k32.CloseHandle(h)


def same_process(pid: int, started: int | None) -> bool:
    """Da li PID i dalje pripada istom procesu (živ, isto vreme pokretanja)."""
    return started is not None and start_time(pid) == started
