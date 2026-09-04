r"""
Unit tests for minikv.server (Module 3).

Created: 2026-09-04 (Module 3)

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

    def test_protocol_error_is_reported_and_the_connection_dropped(self):
        # A framing error is different in kind: there is no way to find the next
        # command in the stream, so the reply is followed by a hang-up. The PING
        # behind it is never seen.
        replies = self.converse(b"*1\r\n$xx\r\n*1\r\n$4\r\nPING\r\n")
        self.assertEqual(replies, b"-ERR Protocol error: invalid bulk length\r\n")


if __name__ == "__main__":
    unittest.main()
