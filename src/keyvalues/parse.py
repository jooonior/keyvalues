from __future__ import annotations

import dataclasses
import functools
import re
import typing
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Self,
    TextIO,
    TypedDict,
    Unpack,
)

from . import utils

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator


class TokenTag(utils.RegexEnum):
    # fmt: off
    SPACE   = r'\n|[^\S\n]+'
    COMMENT = r'//(.*)'
    PLAIN   = r'[^\s"{}]+|[{}]'
    QUOTED  = r'"((?:\\"|[^"])*)"?'
    EOF     = r'^$'
    # fmt: on


class TokenRole(utils.AutoIntEnum):
    KEY = ()
    VALUE = ()
    CONDITION = ()
    OPEN = ()
    CLOSE = ()


class TokenFields(TypedDict, total=False):
    data: str
    tag: int
    flags: int
    depth: int
    role: int | None
    filename: str | None
    line: str
    lineno: int
    start: int
    data_start: int
    end: int
    prev: Token | None
    next: Token | None


@dataclass(slots=True)
class Token:
    data: str
    tag: int = None  # type: ignore[assignment]
    flags: int = 0
    depth: int = -1
    role: int | None = None
    filename: str | None = None
    line: str = None  # type: ignore[assignment]
    lineno: int = 0
    start: int = 0
    data_start: int = 0
    end: int = None  # type: ignore[assignment]
    prev: Token | None = None
    next: Token | None = None

    def __post_init__(self) -> None:
        if self.tag is None:
            match = TokenTag.fullmatch(self.data)  # type: ignore[unreachable]
            if not match:
                errmsg = f"cannot infer tag of {self.data!r}"
                raise ValueError(errmsg)
            self.tag = TokenTag[match.lastgroup]
            self.data = match[match.lastgroup]

        if self.line is None:
            self.line = self.data  # type: ignore[unreachable]

        if self.end is None:
            self.end = self.start + len(self.data)  # type: ignore[unreachable]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Token):
            return False

        return self.data == other.data and self.tag is other.tag

    def __str__(self) -> str:
        match self.tag:
            case TokenTag.QUOTED:
                return f'"{self.data}"'
            case TokenTag.COMMENT:
                return f"//{self.data}"
            case _:
                return self.data

    def __repr__(self) -> str:
        return repr(str(self))

    def clone(self, **kwargs: Unpack[TokenFields]) -> Self:
        return dataclasses.replace(self, **kwargs)

    def __getitem__(self, index: int | slice) -> Self:
        if isinstance(index, slice):
            start, end, _ = index.indices(len(self.data))
        else:
            start = index if index >= 0 else len(self.data) + index
            end = start + 1

        start += self.data_start
        end += self.data_start

        return self.clone(
            data=self.data[index],
            start=start,
            data_start=start,
            end=end,
        )

    def split(self, sep: str) -> Iterator[Token]:
        for match in re.finditer(rf"[^{sep}]+", self.data):
            yield self[match.start() : match.end()]

    def iprev(self) -> Iterator[Token]:
        token = self.prev
        while token is not None:
            yield token
            token = token.prev

    def inext(self) -> Iterator[Token]:
        token = self.next
        while token is not None:
            yield token
            token = token.next

    def drop(self) -> None:
        if self.prev is not None:
            self.prev.next = self.next
        if self.next is not None:
            self.next.prev = self.prev

    @classmethod
    def from_match(
        cls,
        match: re.Match[str],
        **kwargs: Unpack[TokenFields],
    ) -> Self:
        lastgroup = match.lastgroup
        if lastgroup is None:
            errmsg = "match not did not capture any groups"
            raise ValueError(errmsg)

        kwargs.setdefault("data", match[lastgroup])
        kwargs.setdefault("tag", TokenTag[lastgroup])
        kwargs.setdefault("line", match.string)
        kwargs.setdefault("start", match.start())
        kwargs.setdefault("data_start", match.start(lastgroup))
        kwargs.setdefault("end", match.end())

        return cls(**kwargs)


assert typing.get_type_hints(Token) == typing.get_type_hints(TokenFields)


class TokenError(SyntaxError):
    def __init__(self, message: str, token: Token):
        super().__init__(message)
        self.token = token

    # mypy complains because properties have no setters.

    @property
    def filename(self) -> str | None:  # type: ignore[override]
        return self.token.filename

    @property
    def lineno(self) -> int:  # type: ignore[override]
        return self.token.lineno

    @property
    def offset(self) -> int:  # type: ignore[override]
        return self.token.start + 1

    @property
    def text(self) -> str:  # type: ignore[override]
        return self.token.line

    @property
    def end_lineno(self) -> int:  # type: ignore[override]
        return self.token.lineno

    @property
    def end_offset(self) -> int:  # type: ignore[override]
        return self.token.end + 1


class KeyValuesParseError(TokenError):
    pass


def lexer(file: TextIO, filename: str | None = None) -> Iterator[Token]:
    """Split a file-like object `file` into lexical tokens.

    Yields one `Token` for each lexical token in the input file, plus an
    additional EOF token whose `Token.tag` is `TokenTag.EOF`.
    """
    if filename is None:
        filename = getattr(file, "name", None)

        if filename is None:
            filename = f"<{type(file).__module__}.{type(file).__name__}>"

    lines = enumerate(file, 1)
    prev = None  # type: Token | None

    # For the EOF token in case the main `for` loop does not run.
    lineno = 1
    line = ""

    for lineno, line in lines:
        repeat = True
        while repeat:
            repeat = False

            for match in TokenTag.finditer(line):
                token = Token.from_match(
                    match,
                    filename=filename,
                    lineno=lineno,
                )

                token.prev = prev
                if prev is not None:
                    prev.next = token
                prev = token

                # Handle multiline strings.
                if token.tag is TokenTag.QUOTED and (
                    match[0][-1] != '"' or len(match[0]) == 1
                ):
                    parts = [token.data]
                    for lineno, line in lines:  # noqa: B007
                        if quote := re.search(r'(?<!\\)"', line):
                            parts.append(line[: quote.start()])
                            line = line[quote.end() :]
                            break

                        parts.append(line)

                    if len(parts) == 1:
                        token.tag = TokenTag.EOF  # for error handling
                        token.end = token.start + 1

                        errmsg = "unclosed quote"
                        raise KeyValuesParseError(errmsg, token)

                    token.data = "".join(parts)

                    # Next `while repeat` iteration reads rest of the line.
                    repeat = line != ""

                yield token

    yield Token(
        "",
        tag=TokenTag.EOF,
        filename=filename,
        line=line,
        lineno=lineno,
        start=len(line),
    )


def isspace(token: Token) -> bool:
    """Check whether `token` is either whitespace or a comment."""
    return token.tag in {TokenTag.SPACE, TokenTag.COMMENT}


def skipspace(tokens: Iterable[Token]) -> Iterator[Token]:
    """Skip whitespace and comments."""
    return filter(lambda token: not isspace(token), tokens)


def yieldspace(
    tokens: Iterable[Token],
    depth: int | None = None,
) -> Generator[Token, None, Token]:
    """Read until the first `Token` that is neither whitespace nor a comment.

    Yields whitespace and comments and returns the final `Token`.
    If `depth` is given, sets `Token.depth` of yielded and returned tokens.
    Raises `ValueError` when `tokens` is exhausted.
    """
    for token in tokens:
        if depth is not None:
            token.depth = depth

        if isspace(token):
            yield token
        else:
            return token

    errmsg = "no non-space token"
    raise ValueError(errmsg)


def is_condition(token: Token) -> bool:
    """Check whether `token` is a condition."""
    if token.tag is not TokenTag.PLAIN:
        return False

    return -1 < token.data.find("[") < token.data.find("]")


if TYPE_CHECKING:
    ParserFn = Callable[[Iterable[Token], int], Iterator[Token]]
    ParserDecorator = Callable[[ParserFn], ParserFn]


def parser(tokens: Iterable[Token], depth: int) -> Iterator[Token]:
    """Parse `tokens` and set semantic roles of each token.

    Yields from `tokens`, except that each token has its `Token.depth` set and
    semantically relevant tokens have their `Token.role` set.

    Assumes that `tokens` ends with a token whose `Token.tag` is `TokenTag.EOF`.
    """
    tokens = iter(tokens)
    key = yield from yieldspace(tokens, depth)

    while key.tag is not TokenTag.EOF:
        if key.tag is TokenTag.PLAIN and key.data == "}" and depth > 0:
            depth -= 1

            # Don't count closing braces as part of their sections.
            key.depth = depth

            key.role = TokenRole.CLOSE
            yield key

            key = yield from yieldspace(tokens, depth)
            continue

        key.role = TokenRole.KEY
        yield key

        value = yield from yieldspace(tokens, depth)

        if is_condition(value):
            condition = value
            condition.role = TokenRole.CONDITION
            yield condition

            value = yield from yieldspace(tokens, depth)

            if is_condition(value):
                errmsg = "condition instead of value"
                raise KeyValuesParseError(errmsg, value)
        else:
            condition = None

        match value.tag:
            case TokenTag.EOF:
                errmsg = "missing value"
                raise KeyValuesParseError(errmsg, value)

            case TokenTag.PLAIN if value.data == "{":
                value.role = TokenRole.OPEN
                yield value
                depth += 1

                key = yield from yieldspace(tokens, depth)

            case _:
                value.role = TokenRole.VALUE
                yield value

                key = yield from yieldspace(tokens, depth)

                if is_condition(key):
                    condition = key
                    condition.role = TokenRole.CONDITION
                    yield condition

                    key = yield from yieldspace(tokens, depth)

    if depth > 0:
        errmsg = "unclosed section"
        raise KeyValuesParseError(errmsg, key)

    yield key


class Pipeline:
    def __init__(self, parser: ParserFn = parser):
        self.parser = parser

    def parse(
        self,
        file: TextIO,
        filename: str | None = None,
    ) -> Iterable[Token]:
        tokens = lexer(file, filename)
        return self.parser(tokens, 0)

    def add(self, *decorators: ParserDecorator) -> None:
        self.parser = functools.reduce(
            lambda parser, decorator: decorator(parser),
            decorators,
            self.parser,
        )


@dataclass
class ParserIO:
    """Allows reading the input and output of a parser."""

    input: Iterator[Token]
    output: Iterator[Token]

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Token:
        return next(self.output)


class RestartParser(Exception):  # noqa: N818
    pass


def parser_decorator(
    fn: Callable[[ParserIO, int], Iterable[Token]],
) -> ParserDecorator:
    outer = fn

    @functools.wraps(fn)
    def decorator(parser: ParserFn) -> ParserFn:
        inner = parser

        @functools.wraps(parser)
        def decorated(tokens: Iterable[Token], depth: int) -> Iterator[Token]:
            tokens = iter(tokens)

            while True:
                token = None
                io = ParserIO(tokens, inner(tokens, depth))

                try:
                    for token in outer(io, depth):
                        yield token

                    break
                except RestartParser:
                    if token is not None:
                        depth = token.depth

        return decorated

    return decorator


def parse_macros(*macros: str) -> ParserDecorator:
    """Augment parser to parse macros."""
    macros = set(macros)

    @parser_decorator
    def parse_macros(tokens: ParserIO, _depth: int) -> Iterator[Token]:
        for token in tokens.output:
            yield token

            if token.depth > 0:
                continue

            if token.role is not TokenRole.KEY:
                continue

            if token.data not in macros:
                continue

            # Read macro argument from unparsed tokens.
            path = yield from yieldspace(tokens.input, token.depth)

            if path.tag is TokenTag.EOF:
                errmsg = f"missing {token} path"
                raise KeyValuesParseError(errmsg, path)

            if path.data == "":
                errmsg = f"empty {token} path"
                raise KeyValuesParseError(errmsg, path)

            path.role = TokenRole.VALUE
            yield path

            # Restart the parser to parse the next token as a key.
            raise RestartParser

    return parse_macros


def report_errors(
    *,
    treat_empty_root_key_as_error: bool = False,
) -> ParserDecorator:
    """Augment parser to report keys and values not accepted by the SDK parser.

    If `treat_empty_root_key_as_error` is `True`, empty root key raises an error
    instead of being treated as EOF (as done by the SDK parser).

    Should be applied before `parse_macros`.
    """

    @parser_decorator
    def report_errors(tokens: Iterable[Token], _depth: int) -> Iterator[Token]:
        for token in tokens:
            match token.role:
                case TokenRole.KEY if token.data == "":
                    if not treat_empty_root_key_as_error and token.depth == 0:
                        token.tag = TokenTag.EOF
                        yield token
                        return

                    errmsg = "empty key"
                    raise KeyValuesParseError(errmsg, token)

                case TokenRole.VALUE:
                    if token.depth == 0:
                        errmsg = "expected {"
                        raise KeyValuesParseError(errmsg, token)

                    if token.data == "}" and token.tag == TokenTag.PLAIN:
                        errmsg = "} as value"
                        raise KeyValuesParseError(errmsg, token)

            yield token

    return report_errors
