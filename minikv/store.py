r"""
MiniKV - Module 4: The Store

Created: 2026-09-04 (Module 4)

Modules 1-3 built a server that can hear you perfectly and remember nothing.
This module is the dictionary it has been missing - and the lock around it,
which is the part actually worth reading.

The data structure is a Python dict, and that is not a placeholder. Redis is a
hash table too; the interesting engineering in a key-value store is never the
lookup, it is everything around it - concurrency here, expiry in Module 5,
durability in Module 6.

Why the lock, when CPython has a GIL

    A common shortcut says the GIL already makes dict operations atomic, so a
    lock is redundant. That is wrong twice.

    First, it protects the wrong thing. `d[k] = v` is indeed indivisible, but
    the operations a database performs are compound:

        if key not in self._data:      # thread A checks, sees it is missing
            self._data[key] = value    # thread B sets it here; A overwrites

    That is SET NX, and it is a check-then-act race. No amount of per-operation
    atomicity fixes it - the two halves have to be indivisible *together*, which
    is exactly what a lock buys.

    Second, the GIL is a CPython implementation detail, not a language
    guarantee, and as of the free-threaded builds of 3.13+ it is optional. Code
    that leans on it is code that breaks on an interpreter flag.

Why one lock for the whole store

    Every command serialises on `_lock`, so two clients reading different keys
    still take turns. That sounds like the obvious thing to fix - shard the
    keyspace, take a per-key lock - but it is worth noticing that real Redis
    goes further in the opposite direction: it executes commands one at a time,
    on one thread, and is famously fast anyway. In-memory operations are so
    cheap that the lock is held for well under a microsecond, and the contention
    that matters is on the network, not the dict. Module 10 measures this rather
    than assuming it.

    The rule that keeps this honest: the lock is held for the dict operation and
    nothing else. No I/O, no logging, no encoding a reply while holding it.

Keys and values are bytes, never str, all the way down - see protocol.py.
"""

import threading


class KeyValueStore:
    """A dict shared by every client thread, with a lock making it safe.

    Added: 2026-09-04 (Module 4)

    Every public method is one indivisible operation from a caller's point of
    view. That is the whole contract: callers never need their own lock, and
    must never hold one while calling in.
    """

    def __init__(self) -> None:
        self._data: dict[bytes, bytes] = {}

        # A plain Lock, not an RLock. Re-entrancy would let one store method
        # call another and deadlock-proof itself, which sounds convenient and is
        # really a way to build compound operations by accident - the exact bug
        # this class exists to prevent. Methods here touch the dict directly.
        self._lock = threading.Lock()

    def get(self, key: bytes) -> bytes | None:
        """Return the value for `key`, or None if it is not set.

        Added: 2026-09-04 (Module 4)

        None means "no such key", which is why values are bytes: b"" is a value
        a client can legitimately store, and it has to stay distinguishable from
        absence. protocol.bulk_string() carries that distinction onto the wire.
        """
        with self._lock:
            return self._data.get(key)

    def set(
        self,
        key: bytes,
        value: bytes,
        *,
        only_if_missing: bool = False,
        only_if_present: bool = False,
    ) -> bool:
        """Store `value` under `key`. Returns whether it was actually stored.

        Added: 2026-09-04 (Module 4)

        The two flags are SET NX and SET XX, and they are the reason this method
        takes the lock rather than the caller doing `if store.get(k) is None`.
        Between that get() and the set() that followed it, another thread gets
        to run - and the check the caller just made is stale. Here the check and
        the write happen inside one acquisition, so they cannot be split.

        SET NX is how a distributed lock is built, which makes this race the
        kind that hands two clients the same lock and looks fine in testing.
        """
        with self._lock:
            exists = key in self._data
            if only_if_missing and exists:
                return False
            if only_if_present and not exists:
                return False
            self._data[key] = value
            return True

    def delete(self, keys: list[bytes]) -> int:
        """Remove every key in `keys`. Returns how many existed.

        Added: 2026-09-04 (Module 4)

        Variadic and atomic as a group: DEL a b c either happens entirely or
        not at all, and no client can observe the half-deleted state in between.
        """
        with self._lock:
            removed = 0
            for key in keys:
                if self._data.pop(key, None) is not None:
                    removed += 1
            return removed

    def exists(self, keys: list[bytes]) -> int:
        """Count how many of `keys` are set, counting duplicates separately.

        Added: 2026-09-04 (Module 4)

        Duplicates counting twice looks like a bug and is what Redis does:
        EXISTS k k returns 2. It is a count of arguments that hit, not a count
        of distinct keys.
        """
        with self._lock:
            return sum(1 for key in keys if key in self._data)

    def size(self) -> int:
        """How many keys are stored.

        Added: 2026-09-04 (Module 4)
        """
        with self._lock:
            return len(self._data)

    def clear(self) -> int:
        """Drop every key. Returns how many were removed.

        Added: 2026-09-04 (Module 4)
        """
        with self._lock:
            removed = len(self._data)
            self._data.clear()
            return removed
