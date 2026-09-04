r"""
Unit tests for minikv.server (Modules 3-4).

Created: 2026-09-04 (Module 3)
Updated: 2026-09-04 (Module 4 - the keyspace commands)

Two layers, tested two ways.

dispatch() is a pure function - commands in, bytes out, no socket anywhere -
so most of the command set is checked by calling it directly.

The read loop is not pure, and the things most likely to be wrong about it are
things only a real connection exhibits: a batch of pipelined commands answered
in one write, a connection that closes after QUIT, a malformed stream that has
to be reported and then hung up on. Those tests open an actual TCP socket pair
and run handle_client() on its own thread, exactly as the accept loop does.

Run:  python3 -m unittest discover -v
"""

import socket
import threading
import unittest
from unittest import mock

from minikv import protocol, server


class TestDispatch(unittest.TestCase):
    """The command table, exercised without a socket in sight."""

    def setUp(self):
        # server.STORE is module-level state shared by every client thread, so
        # it is also shared by every test. Wiping it before each one keeps the
        # suite order-independent - otherwise a test passes alone and fails in
        # a full run, which is the worst kind of failure to chase.
        server.STORE.clear()
        self.addCleanup(server.STORE.clear)

    def test_ping(self):
        self.assertEqual(server.dispatch([b"PING"]), (b"+PONG\r\n", False))

    def test_ping_with_message(self):
        # PING with an argument replies with a bulk string, not +PONG.
        self.assertEqual(server.dispatch([b"PING", b"hi"]), (b"$2\r\nhi\r\n", False))

    def test_command_names_are_case_insensitive(self):
        for name in (b"ping", b"Ping", b"pInG"):
            with self.subTest(name=name):
                self.assertEqual(server.dispatch([name])[0], protocol.PONG)

    def test_echo(self):
        self.assertEqual(server.dispatch([b"ECHO", b"hello"])[0], b"$5\r\nhello\r\n")

    def test_echo_is_binary_safe(self):
        payload = b"\x00\r\n\xff"
        self.assertEqual(server.dispatch([b"ECHO", payload])[0], protocol.bulk_string(payload))

    def test_wrong_arity(self):
        for command in ([b"ECHO"], [b"ECHO", b"a", b"b"], [b"PING", b"a", b"b"]):
            with self.subTest(command=command):
                reply, close = server.dispatch(command)
                self.assertTrue(reply.startswith(b"-ERR wrong number of arguments"))
                self.assertFalse(close)

    def test_command_docs_is_answered(self):
        # redis-cli sends this before the user types anything; an empty array is
        # the "no introspection here" answer that still lets the client start.
        self.assertEqual(server.dispatch([b"COMMAND", b"DOCS"])[0], protocol.EMPTY_ARRAY)

    def test_quit_asks_for_the_connection_to_close(self):
        self.assertEqual(server.dispatch([b"QUIT"]), (protocol.OK, True))

    def test_unknown_command(self):
        reply, close = server.dispatch([b"FLUX", b"a"])
        self.assertEqual(
            reply,
            b"-ERR unknown command 'FLUX', with args beginning with: 'a', \r\n",
        )
        self.assertFalse(close)

    def test_unknown_command_cannot_inject_a_reply(self):
        # The command name is attacker-controlled and goes back out inside an
        # unframed error reply. A literal CRLF in it would end the error early
        # and let the client read the rest as a second, forged reply.
        reply, _ = server.dispatch([b"EVIL\r\n+INJECTED"])
        self.assertEqual(reply.count(b"\r\n"), 1)
        self.assertTrue(reply.endswith(b"\r\n"))
        self.assertIn(b"\\r\\n", reply)

    def test_long_argument_is_truncated_in_the_error(self):
        reply, _ = server.dispatch([b"FLUX", b"x" * 500])
        self.assertLess(len(reply), 120)
        self.assertIn(b"...", reply)


class TestKeyspaceCommands(unittest.TestCase):
    """SET/GET/DEL/EXISTS as the client sees them, replies and all."""

    def setUp(self):
        server.STORE.clear()
        self.addCleanup(server.STORE.clear)

    def reply(self, *command: bytes) -> bytes:
        return server.dispatch(list(command))[0]

    def test_set_then_get(self):
        self.assertEqual(self.reply(b"SET", b"name", b"minikv"), protocol.OK)
        self.assertEqual(self.reply(b"GET", b"name"), b"$6\r\nminikv\r\n")

    def test_get_missing_key_is_nil_not_an_error(self):
        self.assertEqual(self.reply(b"GET", b"nope"), protocol.NULL_BULK_STRING)

    def test_get_empty_value_is_not_nil(self):
        self.reply(b"SET", b"k", b"")
        self.assertEqual(self.reply(b"GET", b"k"), b"$0\r\n\r\n")

    def test_values_are_binary_safe_end_to_end(self):
        payload = b"\x00\r\n\xff"
        self.reply(b"SET", b"k", payload)
        self.assertEqual(self.reply(b"GET", b"k"), protocol.bulk_string(payload))

    def test_keys_are_case_sensitive_even_though_commands_are_not(self):
        # `set` and `SET` are the same command; `key` and `KEY` are not the
        # same key. Normalising the command name must not touch the arguments.
        self.reply(b"set", b"Key", b"upper")
        self.reply(b"SET", b"key", b"lower")
        self.assertEqual(self.reply(b"GET", b"Key"), b"$5\r\nupper\r\n")
        self.assertEqual(self.reply(b"GET", b"key"), b"$5\r\nlower\r\n")

    def test_set_nx_declines_with_nil(self):
        self.assertEqual(self.reply(b"SET", b"k", b"first", b"NX"), protocol.OK)
        # nil, not +OK: a client using SET NX as a lock has to be able to tell
        # "I stored it" from "someone else already had it".
        self.assertEqual(self.reply(b"SET", b"k", b"second", b"NX"), protocol.NULL_BULK_STRING)
        self.assertEqual(self.reply(b"GET", b"k"), b"$5\r\nfirst\r\n")

    def test_set_xx_declines_a_missing_key(self):
        self.assertEqual(self.reply(b"SET", b"k", b"v", b"xx"), protocol.NULL_BULK_STRING)
        self.reply(b"SET", b"k", b"first")
        self.assertEqual(self.reply(b"SET", b"k", b"second", b"XX"), protocol.OK)

    def test_set_syntax_errors(self):
        for command in ((b"SET", b"k", b"v", b"NX", b"XX"), (b"SET", b"k", b"v", b"EX", b"5")):
            with self.subTest(command=command):
                # EX is a real Redis option MiniKV does not implement yet.
                # Rejecting it beats silently ignoring it and never expiring
                # a key the client believes is temporary.
                self.assertEqual(self.reply(*command), b"-ERR syntax error\r\n")

    def test_del_reports_how_many_existed(self):
        self.reply(b"SET", b"a", b"1")
        self.assertEqual(self.reply(b"DEL", b"a", b"missing"), b":1\r\n")
        self.assertEqual(self.reply(b"GET", b"a"), protocol.NULL_BULK_STRING)

    def test_exists_counts_duplicates(self):
        self.reply(b"SET", b"a", b"1")
        self.assertEqual(self.reply(b"EXISTS", b"a", b"a", b"b"), b":2\r\n")

    def test_dbsize_and_flushall(self):
        self.reply(b"SET", b"a", b"1")
        self.reply(b"SET", b"b", b"2")
        self.assertEqual(self.reply(b"DBSIZE"), b":2\r\n")
        self.assertEqual(self.reply(b"FLUSHALL"), protocol.OK)
        self.assertEqual(self.reply(b"DBSIZE"), b":0\r\n")

    def test_arity_errors(self):
        for command in ((b"SET", b"k"), (b"GET",), (b"GET", b"a", b"b"),
                        (b"DEL",), (b"EXISTS",), (b"DBSIZE", b"x"), (b"FLUSHALL", b"x")):
            with self.subTest(command=command):
                self.assertTrue(self.reply(*command).startswith(b"-ERR wrong number of arguments"))


def tcp_pair() -> tuple[socket.socket, socket.socket]:
    """A connected (client, server) pair over real TCP on the loopback.

    Added: 2026-09-04 (Module 3)

    socket.socketpair() would be shorter, but on Linux it hands back AF_UNIX
    sockets, and handle_client() sets TCP_NODELAY - a TCP option that a Unix
    socket rejects with an OSError. Binding a throwaway listener on port 0 (the
    kernel picks a free port) keeps the test on the same kind of socket the
    server actually serves.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname())
    server_side, _ = listener.accept()
    listener.close()
    return client, server_side


class TestConnection(unittest.TestCase):
    """handle_client() driven over a real socket, as the accept loop drives it."""

    def setUp(self):
        # handle_client logs on connect and disconnect. Silence it so a test
        # run does not look like a server run.
        patcher = mock.patch.object(server, "log")
        self.addCleanup(patcher.stop)
        patcher.start()
        server.STORE.clear()
        self.addCleanup(server.STORE.clear)

    def converse(self, *chunks: bytes) -> bytes:
        """Send chunks, half-close, and read everything the server replies.

        Added: 2026-09-04 (Module 3)
        """
        client, server_side = tcp_pair()
        self.addCleanup(client.close)

        thread = threading.Thread(
            target=server.handle_client,
            args=(server_side, ("127.0.0.1", 0)),
            daemon=True,
        )
        thread.start()

        for chunk in chunks:
            client.sendall(chunk)
        # Half-close: the server sees end-of-stream and finishes, but our
        # direction stays open so we can still read the replies.
        client.shutdown(socket.SHUT_WR)

        received = b""
        while True:
            part = client.recv(4096)
            if not part:
                break
            received += part

        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "client thread did not finish")
        return received

    def test_ping(self):
        self.assertEqual(self.converse(b"*1\r\n$4\r\nPING\r\n"), b"+PONG\r\n")

    def test_inline_ping_for_netcat_users(self):
        self.assertEqual(self.converse(b"PING\r\n"), b"+PONG\r\n")

    def test_command_split_across_two_writes(self):
        # The parser's job, verified through the socket: nothing is answered
        # until the whole command has arrived.
        wire = b"*2\r\n$4\r\nECHO\r\n$5\r\nhello\r\n"
        self.assertEqual(self.converse(wire[:11], wire[11:]), b"$5\r\nhello\r\n")

    def test_pipelined_commands_are_all_answered_in_order(self):
        replies = self.converse(b"*1\r\n$4\r\nPING\r\n*2\r\n$4\r\nECHO\r\n$2\r\nhi\r\n")
        self.assertEqual(replies, b"+PONG\r\n$2\r\nhi\r\n")

    def test_connection_survives_an_unknown_command(self):
        # An application-level error is one command's problem. The stream is
        # still perfectly framed, so the connection stays up.
        replies = self.converse(b"*1\r\n$4\r\nFLUX\r\n*1\r\n$4\r\nPING\r\n")
        self.assertTrue(replies.startswith(b"-ERR unknown command"))
        self.assertTrue(replies.endswith(b"+PONG\r\n"))

    def test_quit_replies_then_closes(self):
        # The commands pipelined behind QUIT are deliberately not answered.
        replies = self.converse(b"*1\r\n$4\r\nQUIT\r\n*1\r\n$4\r\nPING\r\n")
        self.assertEqual(replies, b"+OK\r\n")

    def test_the_store_is_shared_between_connections(self):
        # The reason STORE is module-level and the reason it needs a lock: what
        # one connection writes, a later, separate connection reads.
        self.assertEqual(self.converse(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$2\r\nhi\r\n"), protocol.OK)
        self.assertEqual(self.converse(b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n"), b"$2\r\nhi\r\n")

    def test_inline_set_and_get_in_one_batch(self):
        # What a person typing into netcat gets, pipelined into one packet.
        self.assertEqual(
            self.converse(b"SET a 1\r\nGET a\r\nDEL a\r\nGET a\r\n"),
            b"+OK\r\n$1\r\n1\r\n:1\r\n$-1\r\n",
        )

    def test_protocol_error_is_reported_and_the_connection_dropped(self):
        # A framing error is different in kind: there is no way to find the next
        # command in the stream, so the reply is followed by a hang-up. The PING
        # behind it is never seen.
        replies = self.converse(b"*1\r\n$xx\r\n*1\r\n$4\r\nPING\r\n")
        self.assertEqual(replies, b"-ERR Protocol error: invalid bulk length\r\n")


if __name__ == "__main__":
    unittest.main()
