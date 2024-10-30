from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Iterator,
    )
    from typing import (
        TextIO,
        Unpack,
    )

from . import dbg, utils
from .token import Token, TokenError

if TYPE_CHECKING:
    from .token import TokenKwArgs


class ParsedTokenTag(utils.RegexEnum):
    # fmt: off
    SPACE   = r'\n|[^\S\n]+'
    COMMENT = r'//(.*)'
    PLAIN   = r'[^\s"{}]+|[{}]'
    QUOTED  = r'"((?:\\"|[^"])*)"?'
    EOF     = r'^$'
    # fmt: on


class ParsedTokenRole(utils.AutoIntEnum):
    KEY = ()
    VALUE = ()
    CONDITION = ()
    OPEN = ()
    CLOSE = ()
    MACRO = ()


class ParsedTokenMeta(TypedDict, total=False):
    depth: int
    role: ParsedTokenRole


class ParsedToken(Token[ParsedTokenTag, ParsedTokenMeta]):
    def __init__(
        self,
        data: str,
        **kwargs: Unpack[TokenKwArgs[ParsedTokenTag, ParsedTokenMeta]],
    ):
        if "tag" not in kwargs:
            match = ParsedTokenTag.fullmatch(self.data)
            if match is None:
                errmsg = f"cannot infer tag of {self.data!r}"
                raise ValueError(errmsg)

            lastindex = match.lastindex
            assert lastindex is not None
            data = match[lastindex]
            kwargs["tag"] = lastindex  # type: ignore[typeddict-item]

        super().__init__(data, **kwargs)

    def __repr__(self) -> str:
        role = self.meta.get("role")
        parts = [
            ParsedTokenTag(self.tag).name,
            ParsedTokenRole(role).name if role is not None else "?",
            repr(self.data),
            str(self.meta.get("depth", "?")),
            f"#{id(self):x}",
        ]
        return " ".join(parts)


class ParseError(TokenError):
    pass


def lexer(file: TextIO, filename: str | None = None) -> Iterator[ParsedToken]:
    """Split a file-like object `file` into lexical tokens.

    Yields one `Token` for each lexical token in the input file, plus an
    additional EOF token whose `Token.tag` is `TokenTag.EOF`.
    """
    if filename is None:
        filename = getattr(file, "name", None)

        if filename is None:
            filename = f"<{type(file).__module__}.{type(file).__name__}>"

    lines = enumerate(file, 1)

    # For the EOF token in case the main `for` loop does not run.
    lineno = 1
    line = ""

    for lineno, line in lines:
        repeat = True
        while repeat:
            repeat = False

            for match in ParsedTokenTag.finditer(line):
                token = ParsedToken.from_match(
                    match,
                    filename=filename,
                    lineno=lineno,
                )

                # Handle multi-line strings.
                if token.tag == ParsedTokenTag.QUOTED and (
                    match[0][-1] != '"' or len(match[0]) == 1  # or just '"'
                ):
                    parts = [token.data]
                    for lineno, line in lines:  # noqa: B007
                        quote = re.search(r'(?<!\\)"', line)
                        if quote is not None:
                            parts.append(line[: quote.start()])
                            line = line[quote.end() :]
                            break

                        parts.append(line)

                    if len(parts) == 1:
                        token.tag = ParsedTokenTag.EOF  # for error handling
                        token.end = token.start + 1

                        errmsg = "unclosed quote"
                        raise ParseError(errmsg, token)

                    token.data = "".join(parts)

                    # Next `while repeat` iteration reads rest of the line.
                    repeat = line != ""

                yield token

    yield ParsedToken(
        "",
        tag=ParsedTokenTag.EOF,
        filename=filename,
        line=line,
        lineno=lineno,
        start=len(line),
    )


def isspace(token: ParsedToken) -> bool:
    """Check whether `token` is either whitespace or a comment."""
    return token.tag in {ParsedTokenTag.SPACE, ParsedTokenTag.COMMENT}


def skipspace(tokens: Iterable[ParsedToken]) -> Iterator[ParsedToken]:
    """Skip whitespace and comments."""
    return filter(lambda token: not isspace(token), tokens)


def yieldspace(
    tokens: Iterable[ParsedToken],
    depth: int | None = None,
) -> Generator[ParsedToken, None, ParsedToken]:
    """Read until the first `Token` that is neither whitespace nor a comment.

    Yields whitespace and comments and returns the final `Token`.
    If `depth` is given, sets `Token.depth` of yielded and returned tokens.
    Raises `ValueError` when `tokens` is exhausted.
    """
    for token in tokens:
        if depth is not None:
            token.meta["depth"] = depth

        if isspace(token):
            yield token
        else:
            return token

    errmsg = "no non-space token"
    raise ValueError(errmsg)


def is_condition(token: ParsedToken) -> bool:
    """Check whether `token` is a condition."""
    if token.tag != ParsedTokenTag.PLAIN:
        return False

    return -1 < token.data.find("[") < token.data.find("]")


@dbg.consumer
def basic_parser(
    tokens: Iterable[ParsedToken],
    depth: int,
) -> Iterator[ParsedToken]:
    """Parse `tokens` and assign a semantic metadata to each token.

    Yields from `tokens`, except that each yielded token has its `depth` set
    and semantically relevant tokens have their `role` set too.

    Assumes that `tokens` ends with a token whose `tag` is `ParsedTokenTag.EOF`.
    """
    tokens = iter(tokens)
    key = yield from yieldspace(tokens, depth)

    while True:
        match key.tag:
            case ParsedTokenTag.EOF:
                if depth > 0:
                    errmsg = "unclosed section"
                    raise ParseError(errmsg, key)

                # Don't lose the EOF token!
                yield key
                break

            case ParsedTokenTag.PLAIN if key.data == "}" and depth > 0:
                depth -= 1

                # Don't count closing braces to be inside of their sections.
                key.meta["depth"] = depth

                key.meta["role"] = ParsedTokenRole.CLOSE
                yield key

                key = yield from yieldspace(tokens, depth)
                continue

            case ParsedTokenTag.QUOTED if key.data == "":
                errmsg = "empty key"
                raise ParseError(errmsg, key)

            case _:
                key.meta["role"] = ParsedTokenRole.KEY
                yield key

        value = yield from yieldspace(tokens, depth)

        if is_condition(value):
            condition = value
            condition.meta["role"] = ParsedTokenRole.CONDITION
            yield condition

            value = yield from yieldspace(tokens, depth)

            if is_condition(value):
                errmsg = "condition instead of value"
                raise ParseError(errmsg, value)
        else:
            condition = None

        match value.tag:
            case ParsedTokenTag.EOF:
                errmsg = "missing value"
                raise ParseError(errmsg, value)

            case ParsedTokenTag.PLAIN if value.data == "{":
                value.meta["role"] = ParsedTokenRole.OPEN
                yield value
                depth += 1

                key = yield from yieldspace(tokens, depth)

            case ParsedTokenTag.PLAIN if value.data == "}":
                errmsg = "} as value"
                raise ParseError(errmsg, value)

            case _:
                if depth == 0:
                    errmsg = "value not allowed in root section"
                    raise ParseError(errmsg, value)

                value.meta["role"] = ParsedTokenRole.VALUE
                yield value

                # Read next key and check if it's not a condition instead.

                key = yield from yieldspace(tokens, depth)

                if is_condition(key):
                    condition = key
                    condition.meta["role"] = ParsedTokenRole.CONDITION
                    yield condition

                    key = yield from yieldspace(tokens, depth)


@dataclass(slots=True)
class ParserIO:
    """Allows reading the input and output of a parser."""

    input: Iterator[ParsedToken]
    output: Iterator[ParsedToken]


if TYPE_CHECKING:
    ParserFn = Callable[[Iterable[ParsedToken], int], Iterator[ParsedToken]]
    ParserDecorator = Callable[[ParserFn], ParserFn]


class RestartParser(Exception):  # noqa: N818
    pass


def parser_decorator(
    fn: Callable[[ParserIO, int], Iterator[ParsedToken]],
) -> ParserDecorator:
    outer = fn

    @functools.wraps(fn)
    def decorator(parser: ParserFn) -> ParserFn:
        inner = parser

        @functools.wraps(parser)
        def decorated(
            tokens: Iterable[ParsedToken],
            depth: int,
        ) -> Iterator[ParsedToken]:
            io = ParserIO(iter(tokens), iter([]))

            while True:
                token = None
                io.output = dbg.consumed(inner(io.input, depth), outer)

                try:
                    for token in outer(io, depth):
                        yield token
                except RestartParser:
                    if token is not None:
                        depth = token.meta["depth"]
                        # Opening braces are yielded with outer depth, but
                        # the restarted parser must run with inner depth.
                        if token.meta.get("role") == ParsedTokenRole.OPEN:
                            depth += 1
                    dbg.log(
                        f"restarting {inner.__name__} with depth {depth} "
                        f"after token {token!r} "
                    )
                else:
                    break

        return decorated

    return decorator


def parse_macros(*macros: str) -> ParserDecorator:
    """Augment a parser to parse macros."""
    macros = set(macros)

    def is_macro(token: ParsedToken) -> bool:
        return (
            token.meta["depth"] == 0
            and token.meta.get("role") == ParsedTokenRole.KEY
            and token.data in macros
        )

    @parser_decorator
    def parse_macros(
        tokens: ParserIO,
        _depth: int,
    ) -> Iterator[ParsedToken]:
        for token in tokens.output:
            if not is_macro(token):
                yield token
                continue

            token.meta["role"] = ParsedTokenRole.MACRO
            yield token

            # Read macro argument from unparsed tokens.
            arg = yield from yieldspace(tokens.input, token.meta["depth"])

            if arg.tag == ParsedTokenTag.EOF:
                errmsg = f"missing {token} argument"
                raise ParseError(errmsg, arg)

            if arg.data == "":
                errmsg = f"empty {token} argument"
                raise ParseError(errmsg, arg)

            arg.meta["role"] = ParsedTokenRole.VALUE
            yield arg

            # Restart the parser to parse the next token as a key.
            raise RestartParser

    return parse_macros


class Pipeline:
    def __init__(self, parser: ParserFn = basic_parser):
        self.parser = parser

    def parse(
        self,
        file: TextIO,
        filename: str | None = None,
    ) -> Iterable[ParsedToken]:
        tokens = lexer(file, filename)
        return self.parser(tokens, 0)

    def add(self, *decorators: ParserDecorator) -> None:
        self.parser = functools.reduce(
            lambda parser, decorator: decorator(parser),
            decorators,
            self.parser,
        )
