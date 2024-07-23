from __future__ import annotations

import argparse
import collections
import dataclasses
import enum
import functools
import math
import re
import sys
import traceback
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, TextIO, TypedDict, Union, Unpack

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator


class AutoEnum(enum.IntEnum):
    def __new__(cls) -> Self:
        value = len(cls)
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj


if TYPE_CHECKING:
    # Make mypy think that `RegexEnum` has attributes of `re.Pattern`.

    class RegexEnumMeta(enum.EnumType, re.Pattern[str]):  # type: ignore[misc]
        pass

else:

    class RegexEnumMeta(enum.EnumType):
        def __getattr__(self, name: str) -> Any:
            if self is RegexEnum:
                return getattr(super(), name)

            attr = getattr(self._re, name)
            setattr(self, name, attr)
            return attr


class RegexEnum(enum.IntEnum, metaclass=RegexEnumMeta):
    _ignore_ = "pattern"

    pattern: str

    def __new__(cls, value: str) -> Self:
        index = len(cls) + 1

        obj = int.__new__(cls, index)
        obj._value_ = index
        obj.pattern = value

        return obj

    def __init_subclass__(cls) -> None:
        for member in cls:
            # Convert outermost capturing group into named capturing group.
            open_named_group = f"(?P<{member.name}>"
            regex, n = re.subn(
                r"(?<!\\)\((?!\?)", open_named_group, member.pattern, count=1
            )
            if n == 0:
                # No existing capture group, wrap whole regex in one.
                member.pattern = f"{open_named_group}{regex})"
            else:
                # Wrap regex in non-capturing group to ensure correct
                # interpretation when multiple regexes are joined by "|".
                member.pattern = f"(?:{regex})"

        joined = "|".join(member.pattern for member in cls)
        cls._re = re.compile(joined)  # type: ignore[attr-defined]


class TokenTag(RegexEnum):
    # fmt: off
    SPACE   = r'\n|[^\S\n]+'
    COMMENT = r'//(.*)'
    PLAIN   = r'[^\s"{}]+|[{}]'
    QUOTED  = r'"((?:\\"|[^"])*)"?'
    EOF     = r'^$'
    # fmt: on


class TokenRole(AutoEnum):
    KEY = ()
    VALUE = ()
    CONDITION = ()
    OPEN = ()
    CLOSE = ()


class TokenFields(TypedDict, total=False):
    data: str
    tag: TokenTag
    depth: int
    role: TokenRole | None
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
    tag: TokenTag = None  # type: ignore[assignment]
    depth: int = -1
    role: TokenRole | None = None
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
            self.data = match[self.tag]

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

    def clone(self, **kwargs: Unpack[TokenFields]) -> Token:
        return dataclasses.replace(self, **kwargs)

    def __getitem__(self, index: int | slice) -> Token:
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

    @classmethod
    def from_match(
        cls,
        match: re.Match[str],
        **kwargs: Unpack[TokenFields],
    ) -> Token:
        lastgroup = match.lastgroup
        if lastgroup is None:
            errmsg = "match not did not capture any groups"
            raise ValueError(errmsg)

        tag = TokenTag[lastgroup]

        kwargs.setdefault("data", match[tag])
        kwargs.setdefault("tag", tag)
        kwargs.setdefault("line", match.string)
        kwargs.setdefault("start", match.start())
        kwargs.setdefault("data_start", match.start(tag))
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
    depth: int,
) -> Generator[Token, None, Token]:
    """Read until the first `Token` that is not whitespace nor a comment.

    Yields whitespace and comments and returns the final `Token`.
    Raises `ValueError` when `tokens` is exhausted.
    """
    for token in tokens:
        if isspace(token):
            token.depth = depth
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

            key.depth = depth
            key.role = TokenRole.CLOSE
            yield key

            key = yield from yieldspace(tokens, depth)
            continue

        key.depth = depth
        key.role = TokenRole.KEY
        yield key

        value = yield from yieldspace(tokens, depth)
        value.depth = depth

        if is_condition(value):
            condition = value
            condition.role = TokenRole.CONDITION
            yield condition

            value = yield from yieldspace(tokens, depth)
            value.depth = depth

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
                    condition.depth = depth
                    condition.role = TokenRole.CONDITION
                    yield condition

                    key = yield from yieldspace(tokens, depth)

    if depth > 0:
        errmsg = "unclosed section"
        raise KeyValuesParseError(errmsg, key)

    yield key


if TYPE_CHECKING:
    Parser = Callable[[Iterable[Token], int], Iterator[Token]]
    ParserDecorator = Callable[[Parser], Parser]


def parser_decorator(parser: Parser) -> Callable[[Parser], Parser]:
    """Turn parser into parser decorator."""

    @functools.wraps(parser)
    def decorator(inner_parser: Parser) -> Parser:
        @functools.wraps(inner_parser)
        def decorated(tokens: Iterable[Token], depth: int) -> Iterator[Token]:
            tokens = inner_parser(tokens, depth)
            return parser(tokens, depth)

        return decorated

    return decorator


def parse_macros(*macros: str) -> ParserDecorator:
    """Augment parser to parse macros."""
    macros = set(macros)

    def decorator(parser: Parser) -> Parser:
        @functools.wraps(parser)
        def parse_macros(
            tokens: Iterable[Token], depth: int
        ) -> Iterator[Token]:
            repeat = True
            while repeat:
                repeat = False

                for token in parser(tokens, depth):
                    yield token

                    if (
                        token.depth == 0
                        and token.role is TokenRole.KEY
                        and token.data in macros
                    ):
                        path = yield from yieldspace(tokens, token.depth)

                        if path.tag is TokenTag.EOF:
                            errmsg = f"missing {token} path"
                            raise KeyValuesParseError(errmsg, path)

                        if path.data == "":
                            errmsg = f"empty {token} path"
                            raise KeyValuesParseError(errmsg, path)

                        path.role = TokenRole.VALUE
                        yield path

                        # Restart the parser to parse next key.
                        repeat = True
                        break

        return parse_macros

    return decorator


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


def formatter(
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
        max_consecutive_blank_lines: int | float = math.inf  # type: ignore [no-redef]

    assert isinstance(max_consecutive_blank_lines, (int, float))

    @parser_decorator
    def formatter(tokens: Iterable[Token], _depth: int) -> Iterator[Token]:
        blank_lines = 0
        need_newline = False
        prev_token: Token | None = None
        prev_semantic_token: Token | None = None

        first_line = True

        def nextline(do_indent: bool = True) -> Iterator[Token]:  # noqa: FBT001 FBT002
            nonlocal first_line

            if first_line:
                first_line = False
            else:
                assert isinstance(max_consecutive_blank_lines, int)  # mypy
                count = min(blank_lines, max_consecutive_blank_lines) + 1
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
                        blank_lines += 1

                case TokenTag.COMMENT:
                    if spaces_before_comment < 0:
                        # Preserve whitespace before comments.
                        if blank_lines > 0:
                            yield from nextline(do_indent=False)

                        if (
                            prev_token is not None
                            and prev_token.tag is TokenTag.SPACE
                        ):
                            yield prev_token

                    elif blank_lines > 0:
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
                    blank_lines = 0

                case _:
                    yield token

                    need_newline = False
                    blank_lines = 0

            prev_token = token

            if token.role is not None:
                prev_semantic_token = token

        yield from nextline(do_indent=False)

    return formatter


def autoquote(regex: str = "", *, preserve: bool = False) -> ParserDecorator:
    if regex:
        regex = f"|(?:{regex})"
    regex = r'[\s"{}]' + regex

    pattern = re.compile(regex)

    @parser_decorator
    def autoquote(tokens: Iterable[Token], _depth: int) -> Iterator[Token]:
        for token in tokens:
            if preserve and token.tag is TokenTag.QUOTED:
                yield token
                continue

            if token.role in {TokenRole.KEY, TokenRole.VALUE}:
                if pattern.search(token.data):
                    token.tag = TokenTag.QUOTED
                else:
                    token.tag = TokenTag.PLAIN

            yield token

    return autoquote


def pipeline(parser: Parser, *decorators: Callable[[Parser], Parser]) -> Parser:
    """Apply `decorators` on `parser`."""
    return functools.reduce(
        lambda func, decorator: decorator(func),
        decorators,
        parser,
    )


if TYPE_CHECKING:
    T = typing.TypeVar("T")
    Key = Token
    Condition = Token | None
    Value = Union[Token, T]
    KeyValueTuple = tuple[tuple[Key, Condition], Value[T]]


def loader(
    tokens: Iterable[Token],
    factory: Callable[[Iterable[KeyValueTuple[T]]], T],
) -> T:
    """Build a recursive tree from a parsed tokens.

    Returns an object created by calling `factory` with an iterable of
    `((key, condition), value)` tuples, where `value` is either a `Token` or
    another recursively loaded subsection.

    Assumes that each semantically relevant `Token` in `tokens` has its
    `Token.role` set, except for the last one which represents EOF. Unexpected
    tokens are treated as EOF (error reporting is to be done by the parser).
    """
    return factory(_loader(tokens, factory))


def _loader(
    tokens: Iterable[Token],
    factory: Callable[[Iterable[KeyValueTuple[T]]], T],
) -> Iterator[KeyValueTuple[T]]:
    tokens = skipspace(tokens)
    key = next(tokens)

    while key.role is TokenRole.KEY:
        value = next(tokens)

        if value.role is TokenRole.CONDITION:
            condition = value
            value = next(tokens)
        else:
            condition = None

        match value.role:
            case TokenRole.OPEN:
                section = loader(tokens, factory)
                yield (key, condition), section

                # The recursive call might have read the EOF token.
                nextkey = next(tokens, None)
                if nextkey is None:
                    break

            case TokenRole.VALUE:
                nextkey = next(tokens)
                if nextkey.role is TokenRole.CONDITION:
                    condition = nextkey
                    nextkey = next(tokens)

                yield (key, condition), value

            case _:
                yield (key, condition), factory([])
                break

        key = nextkey


def writer(tokens: Iterable[Token], file: TextIO) -> None:
    """Write `tokens` to `file`."""
    for token in tokens:
        file.write(str(token))


class SubcommandHelpFormatter(argparse.HelpFormatter):
    def _format_action(self, action: argparse.Action) -> str:
        parts = super()._format_action(action)
        if action.nargs == argparse.PARSER:
            # Remove metavar from subparser list and decrease its indentation.
            parts = re.sub(r"^.*\n  |(\n)  ", r"\1", parts)
        return parts


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=SubcommandHelpFormatter)

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="command",
    )

    check_subparser = subparsers.add_parser(
        "check",
        help="check syntax of KeyValues",
        description=(
            "Checks the syntax of input file(s) and reports any errors. "
            "If not input files are specified, reads from stdin instead. "
        ),
    )
    check_subparser.add_argument(
        "files",
        nargs="*",
        type=argparse.FileType("r"),
        default=[sys.stdin],
        help="input file",
        metavar="file",
    )

    file_parser = argparse.ArgumentParser(add_help=False)
    file_parser.add_argument(
        "file",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="input file (default: read from stdin instead)",
    )

    format_parser = argparse.ArgumentParser(add_help=False)
    format_group = format_parser.add_argument_group("output format options")

    # Not lambda because error message shows the function name.
    def indentation(string: str) -> str:
        if string.isspace():
            return string
        return " " * int(string)

    format_group.add_argument(
        "-i",
        "--indent",
        type=indentation,
        default="\t",
        help=(
            "indentation size in spaces per level"
            " (default: use single TAB instead)"
        ),
    )

    format_group.add_argument(
        "-q",
        "--quote",
        type=re.compile,
        help="regex matching tokens to quote (default: preserve quotes)",
    )

    format_group.add_argument(
        "-b",
        "--maxblanks",
        type=int,
        default=1,
        help="maximum number of consecutive blank lines (default: 1)",
    )

    format_group.add_argument(
        "-c",
        "--compact",
        dest="compact",
        action="store_true",
        help="put { on the same line as section key",
    )

    subparsers.add_parser(
        "format",
        help="format KeyValues",
        description=(
            "Formats the input file and prints the formatted output to stdout."
        ),
        parents=[file_parser, format_parser],
    )

    return parser


def make_parser_from_args(parser: Parser, args: argparse.Namespace) -> Parser:
    if args.quote:
        parser = autoquote(args.quote.pattern)(parser)

    return formatter(
        indent=args.indent,
        newline_before_section=not args.compact,
        collapse_empty_sections=args.compact,
        max_consecutive_blank_lines=args.maxblanks,
    )(parser)


def main() -> int:
    args = make_argument_parser().parse_args()

    parse = pipeline(
        parser,
        report_errors(treat_empty_root_key_as_error=False),
        parse_macros("#base", "#include"),
    )

    def parse_file(file: TextIO, parser: Parser) -> Iterator[Token]:
        tokens = lexer(file)
        return parser(tokens, 0)

    try:
        match args.command:
            case "check":
                for file in args.files:
                    tokens = parse_file(file, parse)
                    collections.deque(tokens, maxlen=0)
            case "format":
                parse = make_parser_from_args(parse, args)
                tokens = parse_file(args.file, parse)
                writer(tokens, sys.stdout)

    except TokenError as e:
        traceback.print_exception(e, limit=0)
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
