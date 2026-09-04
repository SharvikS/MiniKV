r"""
Unit tests for minikv.store (Module 4).

Created: 2026-09-04 (Module 4)

Single-threaded tests pin down the semantics - what None means, why EXISTS
counts duplicates, what SET NX returns when it declines.

The threaded tests at the bottom are a different kind of test, and worth being
honest about: a race condition that does not reproduce is not a race condition
that is absent. They cannot prove the locking is correct. What they can do is
make the specific interleaving we are worried about *likely* - many threads
released at once, from a barrier, onto the same key - so that a store without a
lock fails them rather than passing quietly until production.

Run:  python3 -m unittest discover -v
"""

import threading
import unittest

from minikv.store import KeyValueStore


class TestBasics(unittest.TestCase):

    def setUp(self):
        self.store = KeyValueStore()

    def test_missing_key_is_none(self):
        self.assertIsNone(self.store.get(b"nope"))

    def test_set_then_get(self):
        self.assertTrue(self.store.set(b"name", b"minikv"))
        self.assertEqual(self.store.get(b"name"), b"minikv")

    def test_set_overwrites(self):
        self.store.set(b"k", b"first")
        self.store.set(b"k", b"second")
        self.assertEqual(self.store.get(b"k"), b"second")

    def test_empty_value_is_not_absence(self):
        # The distinction the whole nil reply rests on: a key can hold b"",
        # and that must not read back the same as a key that is not there.
        self.store.set(b"k", b"")
        self.assertEqual(self.store.get(b"k"), b"")
        self.assertIsNotNone(self.store.get(b"k"))

    def test_keys_and_values_are_arbitrary_bytes(self):
        key, value = b"\x00\xff\r\n", b"\x80 not utf-8"
        self.store.set(key, value)
        self.assertEqual(self.store.get(key), value)

    def test_delete_returns_how_many_existed(self):
        self.store.set(b"a", b"1")
        self.store.set(b"b", b"2")
        self.assertEqual(self.store.delete([b"a", b"b", b"missing"]), 2)
        self.assertIsNone(self.store.get(b"a"))

    def test_delete_is_idempotent(self):
        self.store.set(b"a", b"1")
        self.assertEqual(self.store.delete([b"a"]), 1)
        self.assertEqual(self.store.delete([b"a"]), 0)

    def test_exists_counts_arguments_not_distinct_keys(self):
        self.store.set(b"a", b"1")
        self.assertEqual(self.store.exists([b"a", b"a", b"b"]), 2)

    def test_size_and_clear(self):
        for i in range(5):
            self.store.set(str(i).encode(), b"v")
        self.assertEqual(self.store.size(), 5)
        self.assertEqual(self.store.clear(), 5)
        self.assertEqual(self.store.size(), 0)


class TestConditionalSet(unittest.TestCase):
    """SET NX and SET XX - the check-then-act the lock exists for."""

    def setUp(self):
        self.store = KeyValueStore()

    def test_nx_sets_a_missing_key(self):
        self.assertTrue(self.store.set(b"k", b"v", only_if_missing=True))
        self.assertEqual(self.store.get(b"k"), b"v")

    def test_nx_declines_an_existing_key_without_changing_it(self):
        self.store.set(b"k", b"original")
        self.assertFalse(self.store.set(b"k", b"new", only_if_missing=True))
        self.assertEqual(self.store.get(b"k"), b"original")

    def test_xx_declines_a_missing_key(self):
        self.assertFalse(self.store.set(b"k", b"v", only_if_present=True))
        self.assertIsNone(self.store.get(b"k"))

    def test_xx_updates_an_existing_key(self):
        self.store.set(b"k", b"original")
        self.assertTrue(self.store.set(b"k", b"new", only_if_present=True))
        self.assertEqual(self.store.get(b"k"), b"new")


class TestConcurrency(unittest.TestCase):
    """Many threads, one store. See this module's docstring on what these prove."""

    THREADS = 32

    def run_concurrently(self, work) -> None:
        """Run `work(i)` on THREADS threads released simultaneously.

        Added: 2026-09-04 (Module 4)

        The barrier is the whole trick. Started threads drift apart by
        milliseconds, which is an eternity - long enough that each one finishes
        before the next begins and no interleaving ever happens. Making them all
        wait and then run at once is what puts them inside the critical section
        together.
        """
        barrier = threading.Barrier(self.THREADS)
        errors = []

        def target(index):
            try:
                barrier.wait()
                work(index)
            except Exception as exc:      # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [
            threading.Thread(target=target, args=(i,), daemon=True)
            for i in range(self.THREADS)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive(), "worker thread did not finish")
        self.assertEqual(errors, [])

    def test_exactly_one_thread_wins_set_nx(self):
        # The distributed-lock scenario. Without the lock, two threads can both
        # see the key missing and both believe they acquired it - which in a
        # real deployment means two workers doing the same job at once.
        store = KeyValueStore()
        winners = []
        winners_lock = threading.Lock()

        def acquire(index):
            if store.set(b"lock", str(index).encode(), only_if_missing=True):
                with winners_lock:
                    winners.append(index)

        self.run_concurrently(acquire)

        self.assertEqual(len(winners), 1)
        # And the winner is the one whose value is actually in the store: the
        # answer the store gave and the state it kept agree.
        self.assertEqual(store.get(b"lock"), str(winners[0]).encode())

    def test_concurrent_writes_to_distinct_keys_all_survive(self):
        # A dict resizing while another thread writes is the classic way to lose
        # an entry in a language without the protection CPython happens to give.
        store = KeyValueStore()

        def fill(index):
            for i in range(100):
                store.set(f"{index}:{i}".encode(), b"v")

        self.run_concurrently(fill)
        self.assertEqual(store.size(), self.THREADS * 100)

    def test_readers_and_writers_never_see_a_torn_value(self):
        # Values are immutable bytes, so a reader can only ever see a value some
        # writer completely stored - never a half-written one. This asserts the
        # property rather than assuming it.
        store = KeyValueStore()
        allowed = {b"short", b"a much longer value than the other one"}
        store.set(b"k", b"short")
        seen = []
        seen_lock = threading.Lock()

        def churn(index):
            for i in range(200):
                if index % 2:
                    store.set(b"k", b"short" if i % 2 else b"a much longer value than the other one")
                else:
                    value = store.get(b"k")
                    with seen_lock:
                        seen.append(value)

        self.run_concurrently(churn)
        self.assertTrue(seen)
        self.assertTrue(set(seen) <= allowed, f"unexpected values: {set(seen) - allowed}")


if __name__ == "__main__":
    unittest.main()
