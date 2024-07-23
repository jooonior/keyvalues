from __future__ import annotations

import argparse
import collections
import dataclasses
import decimal
import enum
import functools
import io
import itertools
import math
import operator
import re
import sys
import traceback
import typing
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Self,
    TextIO,
    TypedDict,
    Union,
    Unpack,
    cast,
    overload,
)


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
    ) -> Self:
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


class KeyValuesPreprocessorError(TokenError):
    pass


def read_balanced(tokens: Iterable[Token], depth: int = 0) -> Iterable[Token]:
    # Yield leading whitespace and comments.
    nonspace = yield from yieldspace(tokens, depth)

    for token in itertools.chain([nonspace], tokens):
        token.depth = depth

        match token.tag:
            case TokenTag.EOF:
                errmsg = "missing }"
                raise KeyValuesPreprocessorError(errmsg, token)

            case TokenTag.PLAIN:
                match token.data:
                    case "{":
                        depth += 1
                    case "}":
                        depth -= 1
                        # Braces are not considered to be inside their section.
                        token.depth = depth

        yield token

        if depth == 0:
            break


def read_directive(tokens: Iterable[Token]) -> Iterator[Token | list[Token]]:
    # No space in topmost section.
    for token in skipspace(tokens):
        match token.tag:
            case TokenTag.EOF:
                errmsg = "unclosed directive"
                raise KeyValuesPreprocessorError(errmsg, token)

            case TokenTag.PLAIN:
                match token.data:
                    case "{":
                        # Capture space an nested sections.
                        yield [token, *read_balanced(tokens, depth=1)]
                        continue

                    case "}":
                        yield token
                        break

        yield token


def yield_directive(*tokens: Token) -> Iterator[Token]:
    yield Token("{", tag=TokenTag.PLAIN)
    yield from tokens
    yield Token("}", tag=TokenTag.PLAIN)


@dataclass
class Definition:
    params: list[Token]
    content: list[Token]
    arity: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.arity = len(self.params)


@dataclass(slots=True)
class Directive:
    begin: Token
    content: list[Token | list[Token]]
    end: Token

    @overload
    def __getitem__(self, index: int) -> Token | list[Token]: ...

    @overload
    def __getitem__(self, index: slice) -> list[Token | list[Token]]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self.content[index]

    def __len__(self) -> int:
        return len(self.content)

    @classmethod
    def read(cls, first: Token, tokens: Iterable[Token]) -> Self:
        content: list[Token | list[Token]] = []

        for token in skipspace(tokens):
            match token.tag:
                case TokenTag.EOF:
                    errmsg = "unclosed directive"
                    raise KeyValuesPreprocessorError(errmsg, token)

                case TokenTag.PLAIN:
                    match token.data:
                        case "{":
                            # Capture space an nested sections.
                            content.append(
                                [token, *read_balanced(tokens, depth=1)]
                            )
                            continue

                        case "}":
                            end = token
                            break

            content.append(token)

        if not content:
            errmsg = "empty directive"
            raise KeyValuesPreprocessorError(errmsg, end)

        return cls(first, content, end)


if TYPE_CHECKING:
    R = typing.TypeVar("R")
    P = typing.ParamSpec("P")


class preprocessor:  # noqa: N801
    def __init__(self, parser: Parser):
        functools.update_wrapper(self, parser)
        self.parser = parser
        self.globals: dict[str, Definition] = {}
        self.locals = collections.ChainMap(self.globals)

    def __call__(self, tokens: Iterable[Token], depth: int) -> Iterator[Token]:
        repeat = True
        while repeat:
            repeat = False
            expand = False

            for token in self.parser(tokens, depth):
                depth = token.depth

                match token.role:
                    case TokenRole.OPEN:
                        self.enter_scope()

                    case TokenRole.CLOSE:
                        self.exit_scope()

                    case TokenRole.KEY:
                        expect_modifier = True

                        if token.tag is TokenTag.PLAIN and token.data == "{":
                            # Directive is read from the unparsed token stream.
                            directive = Directive.read(token, tokens)
                            expanded = self.expand_directive(directive, tokens)

                            # Prepend expanded directive and restart the parser.
                            tokens = itertools.chain(expanded, tokens)
                            repeat = True
                            break

                    case TokenRole.VALUE:
                        expect_modifier = False

                        if expand:
                            token = expand_expressions(
                                token, self.evaluate_definition
                            )

                        expand = False

                    case TokenRole.CONDITION if expect_modifier:
                        expect_modifier = False

                        match token.data:
                            case "$[]":
                                expand = True
                                continue
                            case _:
                                expand = False

                yield token

    def expand_directive(
        self,
        directive: Directive,
        tokens: Iterable[Token],
    ) -> Iterator[Token]:
        first = directive[0]
        if isinstance(first, list):
            errmsg = "invalid directive name"
            raise KeyValuesPreprocessorError(errmsg, first[0])

        match first.data.upper():
            case "COMMENT":
                pass

            case "BEGIN":
                self.enter_scope()

            case "END":
                self.exit_scope()

            case "LOCAL":
                self.store_definition(directive, tokens, self.locals)

            case "GLOBAL":
                self.store_definition(directive, tokens, self.globals)

            case "INCLUDE":
                if len(directive) < 1:
                    errmsg = "missing function name"
                    raise KeyValuesPreprocessorError(errmsg, directive.end)

                name = directive[1]
                if isinstance(name, list):
                    errmsg = "invalid function name"
                    raise KeyValuesPreprocessorError(errmsg, name[0])

                arguments = directive[2:]
                yield from self.expand_definition(name, arguments, 1)

            case _:
                errmsg = f'invalid directive "{first.data}"'
                raise KeyValuesPreprocessorError(errmsg, first)

    def enter_scope(self) -> None:
        self.locals = self.locals.new_child()

    def exit_scope(self) -> None:
        self.locals = self.locals.parents

    def store_definition(
        self,
        directive: Directive,
        tokens: Iterable[Token],
        dest: MutableMapping[str, Definition],
    ) -> None:
        if len(directive) < 2:
            errmsg = "missing function name"
            raise KeyValuesPreprocessorError(errmsg, directive.end)

        name = directive[1]
        if isinstance(name, list):
            errmsg = "invalid function name"
            raise KeyValuesPreprocessorError(errmsg, name[0])

        params = directive[2:]
        for param in params:
            if isinstance(param, list):
                errmsg = "invalid function parameter"
                raise KeyValuesPreprocessorError(errmsg, param[0])

        params = cast(list[Token], params)

        content = [name.clone(depth=0)]
        # Read next condition (optional) and value (or section).
        content.extend(read_balanced(tokens))
        if is_condition(content[-1]):
            content.extend(read_balanced(tokens))

        dest[name.data] = Definition(params, content)

    def expand_definition(
        self,
        name: Token,
        arguments: Sequence[Token | Sequence[Token]],
        min_depth: int = 0,
    ) -> Iterator[Token]:
        definition = self.locals.get(name.data)
        if definition is None:
            errmsg = f'function "{name.data}" not found'
            raise KeyValuesPreprocessorError(errmsg, name)

        if len(arguments) != definition.arity:
            errmsg = (
                f'function "{name.data}" takes {definition.arity}'
                f" arguments, but {len(arguments)} were given"
            )
            e = KeyValuesPreprocessorError(errmsg, name)

            if arguments:
                e.add_note("Function was called with the following arguments:")
                for i, argument in enumerate(arguments, 1):
                    note = io.StringIO()
                    note.write(f"[{i}] ")
                    if isinstance(argument, Token):
                        note.write(str(argument))
                    else:
                        writer(argument, note)
                    e.add_note(note.getvalue())

            raise e

        # Create a new scope for the expanded function.
        yield from yield_directive(Token("BEGIN", tag=TokenTag.PLAIN))

        # Define a nullary local function for each passed argument.
        for param, arg in zip(definition.params, arguments):
            yield from yield_directive(
                Token("LOCAL", tag=TokenTag.PLAIN),
                Token(" ", tag=TokenTag.SPACE),
                param,
            )
            if isinstance(arg, Token):
                yield arg
            else:
                yield from arg

        # Yield only tokens inside a section.
        for token in definition.content:
            if token.depth >= min_depth:
                yield token

        # Close the scope created for the expanded function.
        yield from yield_directive(Token("END", tag=TokenTag.PLAIN))

    def evaluate_definition(
        self,
        name: Token,
        arguments: Sequence[Token],
    ) -> Token:
        tokens = self.expand_definition(name, arguments, 0)

        for token in self(tokens, 0):
            if token.role is TokenRole.VALUE:
                return token

        errmsg = "function did not produce any value"
        raise KeyValuesPreprocessorError(errmsg, name)


class ExpressionTag(RegexEnum):
    # fmt: off
    AND    = r"&&"
    OR     = r"\|\|"

    EQ     = r"=="
    NE     = r"!="
    NOT    = r"!"

    GE     = r">="
    GT     = r">"
    LE     = r"<="
    LT     = r"<"

    PLUS   = r"\+"
    MINUS  = r"-"
    MULT   = r"\*"
    DIVIDE = r"/"
    MODULO = r"%"
    POWER  = r"\^"

    EXPAND = r"\$"

    OPEN   = r"\("
    CLOSE  = r"\)"

    NUMBER = r"[+-]?(?:\.\d+|\d+(?:\.\d*)?)"
    NAME   = r"[^\W\d][\w]*"

    SPACE  = r"\s+"
    ERROR  = r"."
    EOF    = r"$"
    # fmt: on


if TYPE_CHECKING:

    class ExpressionToken(Token):
        tag: ExpressionTag  # type: ignore[assignment]

else:
    ExpressionToken = Token


class KeyValuesExpressionError(TokenError):
    pass


def tokenize_expression(expression: Token) -> Iterator[ExpressionToken]:
    for match in ExpressionTag.finditer(expression.data):
        assert match.lastgroup is not None
        tag = ExpressionTag[match.lastgroup]

        if tag is ExpressionTag.SPACE:
            continue

        token = cast(ExpressionToken, expression[match.start() : match.end()])
        token.tag = tag

        if tag is ExpressionTag.ERROR:
            errmsg = f'unexpected character "{token.data}" in expression'
            raise KeyValuesExpressionError(errmsg, token)

        yield token


def tokenize_arguments(expression: Token) -> Iterator[ExpressionToken]:
    tokens = tokenize_expression(expression)

    for token in tokens:
        match token.tag:
            case ExpressionTag.EOF:
                return

            case ExpressionTag.PLUS | ExpressionTag.MINUS:
                # Join sign and number into a single token.
                number = next(tokens)

                if number.tag is not ExpressionTag.NUMBER:
                    break

                number.data = token.data + number.data
                yield number

            case ExpressionTag.NUMBER | ExpressionTag.NAME:
                yield token

            case _:
                break

    errmsg = f'invalid subtoken "{token.data}" in expanded token'
    raise KeyValuesExpressionError(errmsg, token)


def to_decimal(token: ExpressionToken) -> decimal.Decimal:
    if token.tag is ExpressionTag.NUMBER:
        return decimal.Decimal(token.data)

    errmsg = f'"{token}" cannot be converted to a number'
    raise KeyValuesExpressionError(errmsg, token)


class Operator:
    def __init__(self, op: Callable[..., Any], bp: int, *, infixl: bool = True):
        self.op = op
        self.lbp = bp * 2 + (1 - infixl)
        self.rbp = bp * 2

    def invoke(
        self,
        token: ExpressionToken,
        *args: ExpressionToken,
    ) -> ExpressionToken:
        value = self.op(*map(to_decimal, args))
        if not isinstance(value, decimal.Decimal):
            value = decimal.Decimal(value)

        token = token.clone(data=format(value, "f"))
        token.tag = ExpressionTag.NUMBER
        return token


PREFIX_OPERATORS = {
    ExpressionTag.PLUS: Operator(operator.pos, 9),
    ExpressionTag.MINUS: Operator(operator.neg, 9),
    ExpressionTag.NOT: Operator(operator.not_, 9),
}

INFIX_OPERATORS = {
    ExpressionTag.AND: Operator(lambda x, y: x and y, 0),
    ExpressionTag.OR: Operator(lambda x, y: x or y, 1),
    ExpressionTag.EQ: Operator(operator.eq, 2),
    ExpressionTag.NE: Operator(operator.ne, 2),
    ExpressionTag.LE: Operator(operator.le, 3),
    ExpressionTag.LT: Operator(operator.lt, 3),
    ExpressionTag.GE: Operator(operator.ge, 3),
    ExpressionTag.GT: Operator(operator.gt, 3),
    ExpressionTag.PLUS: Operator(operator.add, 4),
    ExpressionTag.MINUS: Operator(operator.sub, 4),
    ExpressionTag.MULT: Operator(operator.mul, 5),
    ExpressionTag.DIVIDE: Operator(operator.truediv, 5),
    ExpressionTag.MODULO: Operator(operator.mod, 5),
    ExpressionTag.POWER: Operator(operator.pow, 6, infixl=False),
}


T = typing.TypeVar("T")


class Peekable(Generic[T], Iterator[T]):
    def __init__(self, iterable: Iterable[T]):
        self._it = iter(iterable)
        self._next = collections.deque[T]()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        return self._next.popleft() if self._next else next(self._it)

    def __getitem__(self, index: int) -> T:
        if index < 0:
            errmsg = "negative index"
            raise ValueError(errmsg)

        self._load(index + 1)
        return self._next[index]

    def get(self, index: int, default: T | None = None) -> T | None:
        try:
            return self[index]
        except IndexError:
            return default

    def skip(self, count: int) -> None:
        next(itertools.islice(self, count, count), None)

    def _load(self, count: int) -> None:
        loaded = len(self._next)
        if count > loaded:
            self._next.extend(itertools.islice(self._it, count - loaded))


def pratt(
    tokens: Peekable[ExpressionToken],
    expander: Callable[[Token, Sequence[Token]], Token],
    min_bp: int = 0,
    *,
    take_one: bool = False,
    stop_on_expand: bool = False,
) -> ExpressionToken:
    recurse = functools.partial(pratt, tokens, expander)

    token = next(tokens)

    match token.tag:
        case ExpressionTag.NUMBER:
            lhs = token

        case ExpressionTag.OPEN:
            lhs = recurse(0)
            if next(tokens).tag is not ExpressionTag.CLOSE:
                errmsg = "unclosed ("
                raise KeyValuesExpressionError(errmsg, token)

        # Parse prefix operators (including function calls).

        case ExpressionTag.NAME:
            arguments = []
            while True:
                match tokens[0].tag:
                    case ExpressionTag.OPEN:
                        arguments.append(recurse(0, take_one=True))

                    case ExpressionTag.NUMBER | ExpressionTag.NAME:
                        arguments.append(next(tokens))

                    case ExpressionTag.EXPAND if not stop_on_expand:
                        next(tokens)
                        value = recurse(0, take_one=True, stop_on_expand=True)
                        arguments.extend(tokenize_arguments(value))

                    case _:
                        break

            lhs = expander(token, arguments)  # type: ignore[assignment]

            # Set correct tag.
            if match := ExpressionTag.fullmatch(lhs.data):
                assert match.lastgroup is not None
                lhs.tag = ExpressionTag[match.lastgroup]

        case _:
            op = PREFIX_OPERATORS.get(token.tag)
            if op is None:
                errmsg = f'unexpected token "{token.data}" in expression'
                raise KeyValuesExpressionError(errmsg, token)

            rhs = recurse(op.rbp)
            lhs = op.invoke(token, rhs)

    if take_one:
        return lhs

    while True:
        token = tokens[0]

        match token.tag:
            case ExpressionTag.EOF:
                break

            case ExpressionTag.CLOSE:
                break

            # Parse infix operators.

            case _:
                op = INFIX_OPERATORS.get(token.tag)
                if op is None:
                    errmsg = f'expected operator, got "{token}"'
                    raise KeyValuesExpressionError(errmsg, token)

                if op.lbp <= min_bp:
                    break

                next(tokens)

                rhs = recurse(op.rbp)
                lhs = op.invoke(token, lhs, rhs)

    return lhs


def evaluate_expression(
    expression: Token,
    expander: Callable[[Token, Sequence[Token]], Token],
) -> Token:
    try:
        result = pratt(Peekable(tokenize_expression(expression)), expander)
    except TokenError as e:
        errmsg = "error while evaluating expression"
        raise KeyValuesPreprocessorError(errmsg, expression) from e

    return result.clone(tag=TokenTag.QUOTED)


def expand_expressions(
    token: Token,
    expander: Callable[[Token, Sequence[Token]], Token],
) -> Token:
    parts = []

    cursor = 0
    start = 0
    depth = 0

    pattern = re.compile(r"[()]|\$(?:\$|(?=\(|\w))")
    while match := pattern.search(token.data, cursor):
        cursor = match.end()

        match match[0]:
            case "$$" if depth == 0:
                parts.append(token.data[start : cursor - 1])
                start = cursor
                continue

            case "$" if depth == 0:
                parts.append(token.data[start : cursor - 1])
                start = cursor

                if token.data[cursor] == "(":
                    depth = 1
                    cursor += 1
                elif nonword := re.search(r"\W", token.data, cursor):
                    cursor = nonword.start()
                else:
                    cursor = len(token.data)

            case "(" if depth > 0:
                depth += 1

            case ")" if depth > 0:
                depth -= 1

            case _:
                continue

        if depth == 0:
            expression = token[start:cursor]
            parts.append(evaluate_expression(expression, expander).data)
            start = cursor

    parts.append(token.data[start:])
    return token.clone(data="".join(parts), tag=TokenTag.QUOTED)


def pipeline(parser: Parser, *decorators: Callable[[Parser], Parser]) -> Parser:
    """Apply `decorators` on `parser`."""
    return functools.reduce(
        lambda func, decorator: decorator(func),
        decorators,
        parser,
    )


if TYPE_CHECKING:
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

    subparsers.add_parser(
        "expand",
        help="expand KeyValues directives and expressions",
        description=(
            "Expands directives and expressions in the input file and prints"
            " the expanded output to stdout."
        ),
        parents=[file_parser, format_parser],
    )

    return parser


def add_format_parsers(parser: Parser, args: argparse.Namespace) -> Parser:
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
                parse = add_format_parsers(parse, args)
                tokens = parse_file(args.file, parse)
                writer(tokens, sys.stdout)
            case "expand":
                parse = pipeline(
                    parser,
                    preprocessor,
                    report_errors(treat_empty_root_key_as_error=False),
                    parse_macros("#base", "#include"),
                )
                parse = add_format_parsers(parse, args)
                tokens = parse_file(args.file, parse)
                writer(tokens, sys.stdout)

    except TokenError as e:
        traceback.print_exception(e, limit=0)
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
