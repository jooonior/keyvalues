from __future__ import annotations

import collections
import dataclasses
import decimal
import functools
import io
import itertools
import operator
import re
import typing
from collections.abc import (
    Callable,
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
    Union,
    cast,
    overload,
)

from . import utils
from .fmt import writer
from .parse import (
    Token,
    TokenError,
    TokenRole,
    TokenTag,
    is_condition,
    parser_decorator,
    skipspace,
    yieldspace,
)

if TYPE_CHECKING:
    from .parse import ParserFn


class KeyValuesPreprocessorError(TokenError):
    pass


def read_balanced(tokens: Iterable[Token], depth: int = 0) -> Iterable[Token]:
    # Yield leading whitespace and comments.
    nonspace = yield from yieldspace(tokens, depth)

    # Yield tokens until we reach zero depth.
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
    LookupMap = dict[str, Token | LookupMap]  # noqa: F821


class TokenFlags(utils.AutoFlagEnum):
    OVERRIDE = ()


class expand:  # noqa: N801
    def __init__(self, parser: ParserFn):
        functools.update_wrapper(self, parser)

        self.parser = parser
        self.globals: dict[str, Definition] = {}
        self.locals = collections.ChainMap(self.globals)
        self.lookup: list[LookupMap] = [{}]

    def __call__(self, tokens: Iterable[Token], depth: int) -> Iterator[Token]:
        repeat = True
        while repeat:
            repeat = False

            expand = False
            last_key: Token | None = None

            for token in self.parser(tokens, depth):
                depth = token.depth

                match token.role:
                    case TokenRole.OPEN:
                        self.enter_scope()

                        parent = self.lookup[-1]

                        assert last_key is not None
                        child = parent.get(last_key.data)

                        if not isinstance(child, dict):
                            # "." stores the key of this section.
                            child = {".": last_key}
                            parent[last_key.data] = child

                        self.lookup.append(child)

                    case TokenRole.CLOSE:
                        self.exit_scope()

                        self.lookup.pop()
                        assert self.lookup, "unmatched open/close"

                    case TokenRole.KEY:
                        last_key = token
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

                        assert last_key is not None
                        self.lookup[-1][last_key.data] = token

                    case TokenRole.CONDITION if expect_modifier:
                        expect_modifier = False

                        match = re.fullmatch(r"\$\[(.*)\]", token.data)
                        if match is not None:
                            expand = True

                            for flag in match[1].split():
                                flag = flag.upper()
                                if flag == "NOEXPAND":
                                    expand = False
                                else:
                                    assert last_key is not None
                                    last_key.flags |= TokenFlags[flag]

                            continue

                        else:
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
        arguments: Sequence[Token] | None,
    ) -> Token:
        if arguments is not None:
            tokens = self.expand_definition(name, arguments, 0)

            # Function calls in expressions should not affect later lookups.
            lookup = self.lookup.copy()

            for token in self(tokens, 0):
                if token.role is TokenRole.VALUE:
                    self.lookup = lookup
                    return token

            errmsg = "function did not produce any value"
            raise KeyValuesPreprocessorError(errmsg, name)

        # Wihout arguments, evaluate name as relative path.

        levels: list[LookupMap | Token] = self.lookup.copy()  # type: ignore[assignment]

        for key in name.split("/"):
            dots = key.data.count(".")
            if dots == len(key.data):
                for _ in range(dots - 1):
                    levels.pop()
            else:
                parent = levels[-1]

                if isinstance(parent, Token):
                    errmsg = "invalid path"
                    raise KeyValuesPreprocessorError(errmsg, name)

                child = parent.get(key.data)

                if child is None:
                    errmsg = f'key "{key.data}" not found'
                    raise KeyValuesPreprocessorError(errmsg, key)

                levels.append(child)

        value = levels[-1]

        if isinstance(value, Token):
            return value

        # Trailing "." refers to the section name.
        value = value.get(".")
        if value is None:
            errmsg = "path refers to root section"
            raise KeyValuesPreprocessorError(errmsg, name)

        assert isinstance(value, Token)
        return value


class ExpressionTag(utils.RegexEnum):
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
    PATH   = r"\.[/.\w]*"
    NAME   = r"[^\W\d][\w]*"

    SPACE  = r"\s+"
    ERROR  = r".+?"  # only single character, unless matching whole string
    EOF    = r"$"
    # fmt: on


class KeyValuesExpressionError(TokenError):
    pass


def tokenize_expression(expression: Token) -> Iterator[Token]:
    for match in ExpressionTag.finditer(expression.data):
        assert match.lastgroup is not None
        tag = ExpressionTag[match.lastgroup]

        if tag is ExpressionTag.SPACE:
            continue

        token = expression[match.start() : match.end()]
        token.tag = tag

        if tag is ExpressionTag.ERROR:
            errmsg = f'unexpected character "{token.data}" in expression'
            raise KeyValuesExpressionError(errmsg, token)

        yield token


def tokenize_arguments(expression: Token) -> Iterator[Token]:
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


def to_decimal(token: Token) -> decimal.Decimal:
    if token.tag is ExpressionTag.NUMBER:
        return decimal.Decimal(token.data)

    errmsg = f'"{token}" cannot be converted to a number'
    raise KeyValuesExpressionError(errmsg, token)


class Operator:
    def __init__(self, op: Callable[..., Any], bp: int, *, infixl: bool = True):
        self.op = op
        self.lbp = bp * 2 + (1 - infixl)
        self.rbp = bp * 2

    def invoke(self, token: Token, *args: Token) -> Token:
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
    tokens: Peekable[Token],
    expander: Callable[[Token, Sequence[Token] | None], Token],
    min_bp: int = 0,
    *,
    take_one: bool = False,
    stop_on_expand: bool = False,
) -> Token:
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

                    case (
                        ExpressionTag.NUMBER
                        | ExpressionTag.NAME
                        | ExpressionTag.PATH
                    ):
                        arguments.append(next(tokens))

                    case ExpressionTag.EXPAND if not stop_on_expand:
                        next(tokens)
                        value = recurse(0, take_one=True, stop_on_expand=True)
                        arguments.extend(tokenize_arguments(value))

                    case _:
                        break

            lhs = expander(token, arguments)

        case ExpressionTag.PATH:
            lhs = expander(token, None)

        case _:
            op = PREFIX_OPERATORS.get(ExpressionTag(token.tag))
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
                op = INFIX_OPERATORS.get(ExpressionTag(token.tag))
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
    expander: Callable[[Token, Sequence[Token] | None], Token],
) -> Token:
    def expr_expander(name: Token, arguments: Sequence[Token] | None) -> Token:
        token = expander(name, arguments)

        # Set correct tag.
        match = ExpressionTag.fullmatch(token.data)
        assert match is not None, "at least one group should always match"
        assert match.lastgroup is not None
        token.tag = ExpressionTag[match.lastgroup]

        return token

    try:
        result = pratt(Peekable(tokenize_expression(expression)), expr_expander)
    except TokenError as e:
        errmsg = "error while evaluating expression"
        raise KeyValuesPreprocessorError(errmsg, expression) from e

    return result.clone(tag=TokenTag.QUOTED)


def expand_expressions(
    token: Token,
    expander: Callable[[Token, Sequence[Token] | None], Token],
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


if TYPE_CHECKING:
    Key = Token
    Condition = Token | None
    Value = Union[Token, T]
    KeyValueTuple = tuple[tuple[Key, Condition], Value[T]]


def loader(
    tokens: Iterable[Token],
    factory: Callable[[Iterable[KeyValueTuple[T]]], T],
    *,
    pass_braces: bool = False,
) -> T:
    """Build a recursive tree from a parsed tokens.

    Returns an object created by calling `factory` with an iterable of
    `((key, condition), value)` tuples, where `value` is either a `Token` or
    another recursively loaded subsection.

    Assumes that each semantically relevant `Token` in `tokens` has its
    `Token.role` set, except for the last one which represents EOF. Unexpected
    tokens are treated as EOF (error reporting is to be done by the parser).
    """
    return factory(_loader(tokens, factory, pass_braces=pass_braces))  # type: ignore[arg-type]


def _loader(
    tokens: Iterable[Token],
    factory: Callable[[Iterable[KeyValueTuple[T] | Token]], T],
    *,
    pass_braces: bool = False,
) -> Iterator[KeyValueTuple[T] | Token]:
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
                section = _loader(tokens, factory, pass_braces=pass_braces)

                if pass_braces:
                    section = itertools.chain([value], section)

                yield (key, condition), factory(section)

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

    if pass_braces and key.role is TokenRole.CLOSE:
        yield key


@dataclass(slots=True)
class KeyConditionValue(Generic[T]):
    key: Token
    condition: Token | None
    value: Token | T

    @classmethod
    def from_tuple(cls, tuple_: KeyValueTuple[T]) -> Self:
        (key, condition), value = tuple_
        return cls(key, condition, value)


class MergedKeyValues:
    def __init__(
        self,
        loaded: Iterable[KeyValueTuple[MergedKeyValues] | Token],
    ):
        self.items: list[KeyConditionValue[MergedKeyValues]] = []
        self.index: dict[tuple[str, str | None], int] = {}

        loaded = iter(loaded)

        # Peek first item.
        first = next(loaded, None)
        self.open = first if isinstance(first, Token) else None
        self.close = None

        # Put first item back.
        if isinstance(first, tuple):
            loaded = itertools.chain([first], loaded)

        for item in loaded:
            if isinstance(item, Token):
                self.close = item
                break

            self.append(KeyConditionValue.from_tuple(item))

    def append(self, item: KeyConditionValue[MergedKeyValues]) -> None:
        key = item.key
        condition = item.condition
        value = item.value

        index_key = (
            key.data,
            None if condition is None else condition.data,
        )

        if TokenFlags.OVERRIDE & key.flags:
            i = self.index.get(index_key)
            if i is not None:
                old = self.items[i]

                if isinstance(old.value, MergedKeyValues) and isinstance(
                    value, MergedKeyValues
                ):
                    old.value.merge(value)
                else:
                    self.items[i].value = value

                return

        self.index[index_key] = len(self.items)
        self.items.append(item)

    def merge(self, other: MergedKeyValues) -> None:
        for item in other.items:
            self.append(item)

    def __iter__(self) -> Iterator[Token]:
        def yield_token(token: Token) -> Iterator[Token]:
            yield token
            yield from yieldspace(token.inext())

        if self.open is not None:
            yield from yield_token(self.open)

        for item in self.items:
            yield from yield_token(item.key)

            if isinstance(item.value, MergedKeyValues):
                yield from item.value
            else:
                yield from yield_token(item.value)

            if item.condition is not None:
                yield from yield_token(item.condition)

        if self.close is not None:
            yield from yield_token(self.close)


def chain(tokens: Iterable[Token]) -> Iterator[Token]:
    prev: Token | None = None

    for token in tokens:
        token.prev = prev

        if prev is not None:
            prev.next = token

        prev = token

        yield token


@parser_decorator
def merge(tokens: Iterable[Token], _depth: int) -> Iterator[Token]:
    tokens = chain(tokens)

    # Peek first token.
    tokens = iter(tokens)
    first = next(tokens)
    tokens = itertools.chain([first], tokens)

    merged = loader(tokens, MergedKeyValues, pass_braces=True)

    yield first

    if first.tag is not TokenTag.EOF:
        yield from yieldspace(first.inext())

    yield from merged
