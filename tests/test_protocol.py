r"""
Unit tests for minikv.protocol (Module 3).

Created: 2026-09-04 (Module 3)

The encoder tests are spot checks - the functions are three lines each and the
only thing worth pinning down is the exact bytes on the wire.

The parser tests are the real ones, and they are mostly about *chunk
boundaries*. A parser that is only ever fed whole commands looks perfect and
falls apart the first time a command straddles two packets, so the tests below
deliberately feed it one byte at a time, split it in every possible place, and
glue commands together.

Run:  python3 -m unittest discover -v
"""

import unittest

from minikv import protocol
from minikv.protocol import ProtocolError, RESPParser


def parse_all(*chunks: bytes) -> list[list[bytes]]:
    """Feed chunks to one parser in order and collect every command produced.

    Added: 2026-09-04 (Module 3)
    """
    parser = RESPParser()
    commands = []
    for chunk in chunks:
        parser.feed(chunk)
        commands.extend(parser.commands())
    return commands


class TestEncoding(unittest.TestCase):
    """Python values -> the exact bytes RESP calls for."""

    def test_simple_string(self):
        self.assertEqual(protocol.simple_string("OK"), b"+OK\r\n")
        self.assertEqual(protocol.OK, b"+OK\r\n")
        self.assertEqual(protocol.PONG, b"+PONG\r\n")

    def test_simple_string_rejects_newlines(self):
        # An unframed reply containing CRLF would end early and desynchronise
        # the client's parser, so the encoder refuses to produce one.
        with self.assertRaises(ValueError):
            protocol.simple_string("two\r\nlines")

    def test_error(self):
        self.assertEqual(protocol.error("ERR nope"), b"-ERR nope\r\n")

    def test_integer(self):
        self.assertEqual(protocol.integer(0), b":0\r\n")
        self.assertEqual(protocol.integer(42), b":42\r\n")
        self.assertEqual(protocol.integer(-1), b":-1\r\n")

    def test_bulk_string(self):
        self.assertEqual(protocol.bulk_string(b"hello"), b"$5\r\nhello\r\n")
        self.assertEqual(protocol.bulk_string("hello"), b"$5\r\nhello\r\n")
        self.assertEqual(protocol.bulk_string(b""), b"$0\r\n\r\n")

    def test_bulk_string_is_binary_safe(self):
        # The length prefix means the payload is never scanned, so bytes that
        # would wreck a delimiter-based protocol travel through untouched.
        payload = b"line\r\nline\x00\xff"
        self.assertEqual(
            protocol.bulk_string(payload),
            b"$12\r\n" + payload + b"\r\n",
        )

    def test_bulk_string_length_is_bytes_not_characters(self):
        # "e" with a combining acute is two characters and three UTF-8 bytes.
        # RESP counts bytes; getting this wrong truncates every non-ASCII value.
        self.assertEqual(protocol.bulk_string("é"), b"$2\r\n\xc3\xa9\r\n")

    def test_null_bulk_string_differs_from_empty(self):
        # This distinction is the whole reason GET can say "no such key".
        self.assertEqual(protocol.bulk_string(None), b"$-1\r\n")
        self.assertNotEqual(protocol.bulk_string(None), protocol.bulk_string(b""))

    def test_array(self):
        encoded = protocol.array([
            protocol.bulk_string(b"ECHO"),
            protocol.bulk_string(b"hi"),
        ])
        self.assertEqual(encoded, b"*2\r\n$4\r\nECHO\r\n$2\r\nhi\r\n")

    def test_array_is_heterogeneous(self):
        encoded = protocol.array([protocol.bulk_string(b"n"), protocol.integer(1)])
        self.assertEqual(encoded, b"*2\r\n$1\r\nn\r\n:1\r\n")

    def test_null_array_differs_from_empty_array(self):
        self.assertEqual(protocol.array(None), b"*-1\r\n")
        self.assertEqual(protocol.array([]), b"*0\r\n")

    def test_encoded_command_round_trips_through_the_parser(self):
        # The two halves of the module agree with each other, which is the one
        # property neither half can establish alone.
        wire = protocol.array([protocol.bulk_string(p) for p in (b"SET", b"k", b"v")])
        self.assertEqual(parse_all(wire), [[b"SET", b"k", b"v"]])


class TestParsingWholeCommands(unittest.TestCase):
    """The easy case: each command arrives complete and alone."""

    def test_single_command(self):
        self.assertEqual(
            parse_all(b"*2\r\n$4\r\nECHO\r\n$5\r\nhello\r\n"),
            [[b"ECHO", b"hello"]],
        )

    def test_one_argument_command(self):
        self.assertEqual(parse_all(b"*1\r\n$4\r\nPING\r\n"), [[b"PING"]])

    def test_empty_bulk_argument(self):
        # SET key "" is a real thing to do; a zero-length payload is not a bug.
        self.assertEqual(
            parse_all(b"*2\r\n$3\r\nGET\r\n$0\r\n\r\n"),
            [[b"GET", b""]],
        )

    def test_binary_argument_survives(self):
        payload = b"\x00\r\n\xff binary"
        wire = b"*2\r\n$3\r\nSET\r\n$" + str(len(payload)).encode() + b"\r\n" + payload + b"\r\n"
        self.assertEqual(parse_all(wire), [[b"SET", payload]])

    def test_empty_and_null_arrays_are_ignored(self):
        # Not an error, and crucially not a wedge: the header is consumed, so a
        # following command still parses.
        self.assertEqual(parse_all(b"*0\r\n*-1\r\n*1\r\n$4\r\nPING\r\n"), [[b"PING"]])


class TestChunkBoundaries(unittest.TestCase):
    """The case that actually matters: recv() splits wherever it likes."""

    WIRE = b"*3\r\n$3\r\nSET\r\n$4\r\nname\r\n$6\r\nminikv\r\n"
    EXPECTED = [b"SET", b"name", b"minikv"]

    def test_split_at_every_possible_offset(self):
        # Exhaustive rather than representative: any off-by-one in the
        # completeness checks shows up as exactly one failing split point.
        for cut in range(len(self.WIRE) + 1):
            with self.subTest(cut=cut):
                self.assertEqual(
                    parse_all(self.WIRE[:cut], self.WIRE[cut:]),
                    [self.EXPECTED],
                )

    def test_one_byte_at_a_time(self):
        chunks = [self.WIRE[i:i + 1] for i in range(len(self.WIRE))]
        self.assertEqual(parse_all(*chunks), [self.EXPECTED])

    def test_nothing_is_emitted_before_the_command_is_complete(self):
        # The dangerous failure mode is not "crashes on a partial command", it
        # is "acts on a partial command", so assert on the silence.
        parser = RESPParser()
        parser.feed(self.WIRE[:-1])
        self.assertEqual(list(parser.commands()), [])
        parser.feed(self.WIRE[-1:])
        self.assertEqual(list(parser.commands()), [self.EXPECTED])

    def test_split_inside_the_trailing_crlf(self):
        # The payload has fully arrived but its terminator has not. A parser
        # that only checked the payload length would emit here.
        parser = RESPParser()
        parser.feed(b"*1\r\n$4\r\nPING\r")
        self.assertEqual(list(parser.commands()), [])
        parser.feed(b"\n")
        self.assertEqual(list(parser.commands()), [[b"PING"]])

    def test_pipelined_commands_in_one_chunk(self):
        # Several commands in one read is the same case as half a command in
        # one read, which is why pipelining needs no extra code.
        self.assertEqual(
            parse_all(b"*1\r\n$4\r\nPING\r\n" * 3),
            [[b"PING"], [b"PING"], [b"PING"]],
        )

    def test_pipelined_commands_split_mid_stream(self):
        stream = b"*1\r\n$4\r\nPING\r\n*2\r\n$4\r\nECHO\r\n$2\r\nhi\r\n"
        self.assertEqual(
            parse_all(stream[:9], stream[9:22], stream[22:]),
            [[b"PING"], [b"ECHO", b"hi"]],
        )

    def test_leftovers_persist_across_drains(self):
        # commands() is called between feeds; the half-command it declines to
        # parse must still be there afterwards.
        parser = RESPParser()
        parser.feed(b"*1\r\n$4\r\nPING\r\n*1\r\n$4\r\n")
        self.assertEqual(list(parser.commands()), [[b"PING"]])
        self.assertEqual(list(parser.commands()), [])
        parser.feed(b"PING\r\n")
        self.assertEqual(list(parser.commands()), [[b"PING"]])


class TestInlineCommands(unittest.TestCase):
    """The telnet/netcat form, so a human can drive the server by hand."""

    def test_inline_command(self):
        self.assertEqual(parse_all(b"PING\r\n"), [[b"PING"]])

    def test_inline_command_with_arguments(self):
        self.assertEqual(parse_all(b"SET name minikv\r\n"), [[b"SET", b"name", b"minikv"]])

    def test_inline_extra_whitespace_is_collapsed(self):
        self.assertEqual(parse_all(b"  ECHO   hi  \r\n"), [[b"ECHO", b"hi"]])

    def test_bare_newline_from_telnet(self):
        # Some clients send LF without CR. split() drops the stray \r either way.
        self.assertEqual(parse_all(b"PING\n"), [[b"PING"]])

    def test_blank_line_is_ignored(self):
        self.assertEqual(parse_all(b"\r\nPING\r\n"), [[b"PING"]])

    def test_inline_waits_for_its_terminator(self):
        parser = RESPParser()
        parser.feed(b"PIN")
        self.assertEqual(list(parser.commands()), [])
        parser.feed(b"G\r\n")
        self.assertEqual(list(parser.commands()), [[b"PING"]])


class TestProtocolErrors(unittest.TestCase):
    """Malformed input must fail loudly, not be guessed at."""

    def assert_rejects(self, wire: bytes):
        with self.assertRaises(ProtocolError):
            parse_all(wire)

    def test_non_numeric_array_count(self):
        self.assert_rejects(b"*abc\r\n")

    def test_non_numeric_bulk_length(self):
        self.assert_rejects(b"*1\r\n$abc\r\n")

    def test_array_element_is_not_a_bulk_string(self):
        # Arguments are always bulk strings; an integer here means the client is
        # broken, and guessing at its intent would corrupt the framing.
        self.assert_rejects(b"*1\r\n:1\r\n")

    def test_null_bulk_string_is_not_a_valid_argument(self):
        self.assert_rejects(b"*1\r\n$-1\r\n")

    def test_bulk_string_not_terminated_by_crlf(self):
        # Length says 4, and the four bytes are there, but they are followed by
        # junk instead of CRLF - the length was a lie.
        self.assert_rejects(b"*1\r\n$4\r\nPINGxx\r\n")

    def test_lengths_with_padding_or_separators_are_rejected(self):
        # int() would happily accept all of these. RESP does not.
        for header in (b"*1\r\n$ 4\r\n", b"*1\r\n$+4\r\n", b"*1\r\n$1_0\r\n", b"*1\r\n$\r\n"):
            with self.subTest(header=header):
                self.assert_rejects(header)

    def test_oversized_bulk_length_is_refused_before_allocating(self):
        # The point of the limit: this is rejected on the header alone, without
        # waiting for (or reserving room for) a gigabyte of payload.
        self.assert_rejects(b"*1\r\n$" + str(protocol.MAX_BULK_LENGTH + 1).encode() + b"\r\n")

    def test_oversized_array_count_is_refused_before_allocating(self):
        self.assert_rejects(b"*" + str(protocol.MAX_ARRAY_LENGTH + 1).encode() + b"\r\n")

    def test_endless_line_without_crlf_is_refused(self):
        parser = RESPParser()
        parser.feed(b"x" * (protocol.MAX_LINE_LENGTH + 1))
        with self.assertRaises(ProtocolError):
            list(parser.commands())

    def test_a_long_but_legal_line_is_not_refused(self):
        # The limit must not trip on data that simply has not finished arriving.
        parser = RESPParser()
        parser.feed(b"x" * (protocol.MAX_LINE_LENGTH - 1))
        self.assertEqual(list(parser.commands()), [])


if __name__ == "__main__":
    unittest.main()
