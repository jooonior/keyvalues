from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, TextIO

from .parse import (
    Token,
    TokenRole,
    TokenTag,
    is_condition,
    parser_decorator,
    skipspace,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from .parse import ParserDecorator


def normalize_whitespace(
    *,
    indent: str | int = "\t",
    newline_before_section: bool = True,
    collapse_empty_sections: bool = False,
    max_consecutive_blank_lines: int | None = 1,
    spaces_before_comment: int = 2,
    comment_starts_with_space: bool = True,
) -> ParserDecorator:
    if isinstance(indent, int):
        indent = " " * indent
    elif not indent.isspace():
        errmsg = "invalid indentation string"
        raise ValueError(errmsg)

    if max_consecutive_blank_lines is None:
        max_consecutive_newlines: int | float = math.inf
    else:
        max_consecutive_newlines = max_consecutive_blank_lines + 1

    @parser_decorator
    def formatter(tokens: Iterable[Token], _depth: int) -> Iterator[Token]:
        newlines = 0
        need_newline = False
        prev_token: Token | None = None
        prev_semantic_token: Token | None = None

        first_line = True

        def nextline(do_indent: bool = True) -> Iterator[Token]:  # noqa: FBT001 FBT002
            nonlocal first_line

            if first_line:
                first_line = False
            else:
                # Always an `int` because `math.inf` is never the smaller value.
                count: int = min(newlines, max_consecutive_newlines)  # type: ignore[assignment]
                count = max(count, 1)
                for _ in range(count):
                    yield Token("\n", tag=TokenTag.SPACE)

            if do_indent and token.depth > 0:
                assert isinstance(indent, str)  # mypy
                yield Token(indent * token.depth, tag=TokenTag.SPACE)

        def separator(width: int = 1) -> Iterator[Token]:
            if need_newline:
                yield from nextline()
            else:
                yield Token(" " * width)

        for token in tokens:
            # Generate whitespace.
            match token.role:
                case TokenRole.KEY:
                    yield from nextline()

                case TokenRole.VALUE:
                    yield from separator()

                case TokenRole.CONDITION:
                    yield from separator()

                case TokenRole.OPEN:
                    if newline_before_section:
                        yield from nextline()
                    else:
                        yield from separator()

                case TokenRole.CLOSE:
                    if not collapse_empty_sections or (
                        prev_semantic_token is None
                        or prev_semantic_token.role is not TokenRole.OPEN
                    ):
                        yield from nextline()

            match token.tag:
                case TokenTag.SPACE:
                    if token.data == "\n":
                        newlines += 1

                case TokenTag.COMMENT:
                    if spaces_before_comment < 0:
                        # Preserve whitespace before comments.
                        if newlines > 0:
                            yield from nextline(do_indent=False)

                        if (
                            prev_token is not None
                            and prev_token.tag is TokenTag.SPACE
                        ):
                            yield prev_token

                    elif newlines > 0:
                        # Comment lies on its own line.
                        yield from nextline()

                    else:
                        # Comment lies on the same line as other tokens.
                        yield from separator(spaces_before_comment)

                    if comment_starts_with_space:
                        token.data = token.data.strip()
                        # Preserve comments like "//----" and such.
                        if token.data[0].isalnum():
                            token.data = f" {token.data}"
                    else:
                        token.data = token.data.rstrip()

                    yield token

                    need_newline = True
                    newlines = 0

                case _:
                    yield token

                    need_newline = False
                    newlines = 0

            prev_token = token

            if token.role is not None:
                prev_semantic_token = token

        yield from nextline(do_indent=False)

    return formatter


def autoquote(regex: str = "", *, preserve: bool = False) -> ParserDecorator:
    """Normalize quoting of keys and values.

    Removes quotes from keys and values which do not need to be quoted.
    If `regex` is given, keys and values matching it are unconditionally quoted.
    If `preserve` is `True`, quotes are only added, not removed.
    """
    if regex:
        regex = f"|(?:{regex})"
    regex = r'^$|[\s"{}]' + regex

    pattern = re.compile(regex)

    @parser_decorator
    def autoquote(tokens: Iterable[Token], _depth: int) -> Iterator[Token]:
        for token in tokens:
            if preserve and token.tag is TokenTag.QUOTED:
                yield token
                continue

            match token.role:
                case TokenRole.VALUE:
                    if pattern.search(token.data):
                        token.tag = TokenTag.QUOTED
                    else:
                        token.tag = TokenTag.PLAIN
                        # Put quotes back if it would be parsed as a condition.
                        if is_condition(token):
                            token.tag = TokenTag.QUOTED

                case TokenRole.KEY:
                    if pattern.search(token.data):
                        token.tag = TokenTag.QUOTED
                    else:
                        token.tag = TokenTag.PLAIN

            yield token

    return autoquote


@parser_decorator
def minify(tokens: Iterable[Token], _depth: int) -> Iterator[Token]:
    """Remove comments and all but necessary whitespace."""
    need_space = False

    for token in skipspace(tokens):
        if token.tag is TokenTag.PLAIN and token.role in {
            TokenRole.KEY,
            TokenRole.VALUE,
            TokenRole.CONDITION,
        }:
            if need_space:
                yield Token(" ", tag=TokenTag.SPACE)

            need_space = True

        else:
            need_space = False

        yield token


def writer(tokens: Iterable[Token], file: TextIO) -> None:
    """Write `tokens` to `file`."""
    for token in tokens:
        file.write(str(token))
