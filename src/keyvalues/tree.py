from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
    assert_never,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

from .parse import ParsedToken, ParsedTokenRole, ParsedTokenTag
from .token import TokenError
from .utils import CaseInsensitiveDict

if TYPE_CHECKING:
    from .token import AnyToken


T = TypeVar("T")


@dataclass(slots=True)
class Entry(Generic[T]):
    key: ParsedToken
    condition: ParsedToken | None
    value: ParsedToken | Section[T]


@dataclass(slots=True)
class Section(Generic[T]):
    open: ParsedToken | None
    children: T
    close: ParsedToken | None


class KeyValues:
    def __init__(
        self,
        parent: Section[KeyValues] | None = None,
        key: ParsedToken | None = None,
        condition: ParsedToken | None = None,
    ) -> None:
        self.parent = parent
        self.key = key
        self.condition = condition

        self._children: list[Entry[KeyValues] | None] = []
        self._by_key: CaseInsensitiveDict[int] = CaseInsensitiveDict()
        self._by_key_and_condition = self._by_key.copy()

    def __iter__(self) -> Iterator[Entry[KeyValues]]:
        for entry in self._children:
            if entry is not None:
                yield entry

    def at(self, index: int) -> Entry[KeyValues]:
        entry = self._children[index]
        if entry is None:
            errmsg = "item has been deleted"
            raise IndexError(errmsg)
        return entry

    def _lookup(
        self,
        key: AnyToken,
        condition: AnyToken | None,
    ) -> tuple[str, str]:
        by_key = key.data

        by_key_and_condition = f"{key.data}\0"
        if condition is not None:
            by_key_and_condition += condition.data

        return by_key, by_key_and_condition

    def get(self, key: AnyToken | str) -> Entry[KeyValues] | None:
        if not isinstance(key, str):
            key = key.data

        index = self._by_key.get(key)
        if index is None:
            return None

        entry = self._children[index]
        assert entry is not None, "accessing deleted entry"

        return entry

    def delete(self, key: AnyToken | str) -> bool:
        if not isinstance(key, str):
            key = key.data

        index = self._by_key.get(key)
        if index is None:
            return False

        entry = self.at(index)
        self._children[index] = None

        by_key, by_key_and_condition = self._lookup(entry.key, entry.condition)
        del self._by_key[by_key]
        del self._by_key_and_condition[by_key_and_condition]

        return True

    def walk(self, path: AnyToken) -> ParsedToken | KeyValues:
        section = self

        keys = path.split("/")
        for key in keys:
            dots = key.data.count(".")

            if dots == len(key.data):
                for _ in range(dots - 1):
                    if section.parent is None:
                        errmsg = "path leads outside of root section"
                        raise TokenError(errmsg, key)

                    section = section.parent.children

            else:
                child = section.get(key)
                if child is None:
                    errmsg = f'path leads to non-existing key "{key.data}"'
                    raise TokenError(errmsg, key)

                value = child.value
                if isinstance(value, ParsedToken):
                    # Only OK if this is the last key.
                    if next(keys, None) is not None:
                        errmsg = f'path segment "{key.data}" leads to a value'
                        raise TokenError(errmsg, key)

                    return value

                section = value.children

        return section

    def append(self, child: Entry[KeyValues]) -> int:
        key = child.key.data
        condition = "" if child.condition is None else child.condition.data

        index = len(self._children)
        self._by_key[key] = index
        self._by_key_and_condition[f"{key}\0{condition}"] = index

        self._children.append(child)
        return index

    def insert(self, child: Entry[KeyValues]) -> int:
        _, by_key_and_condition = self._lookup(child.key, child.condition)
        index = self._by_key_and_condition.get(by_key_and_condition)

        if index is None:
            return self.append(child)

        new = child
        old = self.at(index)
        if isinstance(new.value, Section) and isinstance(old.value, Section):
            old.value.children.merge(new.value.children)
        else:
            old.value = new.value

        return index

    def merge(self, other: KeyValues) -> None:
        for child in other:
            self.insert(child)

    def tokens(self) -> Iterator[ParsedToken]:
        for child in self:
            yield child.key

            if child.condition is not None:
                yield child.condition

            if isinstance(child.value, ParsedToken):
                yield child.value

            else:
                assert child.value.open is not None
                yield child.value.open

                yield from child.value.children.tokens()

                assert child.value.close is not None
                yield child.value.close


class Builder:
    """Build `KeyValues` tree by pushing individual tokens.

    Each `KeyValue` is inserted only when the next key is pushed.
    """

    def __init__(self) -> None:
        self._key: ParsedToken | None = None
        self._condition: ParsedToken | None = None
        self._value: ParsedToken | None = None
        self._stack = [Section(None, KeyValues(), None)]
        self._merge = True
        self._merge_next = self._merge

    def configure(
        self,
        *,
        merge: bool | None = None,
    ) -> None:
        if merge is not None:
            self._merge = merge
            self._merge_next = merge

    def get(self) -> KeyValues:
        """Get the children of the inner-most currently open section."""
        return self._stack[-1].children

    def commit(self) -> None:
        """Insert the current `KeyValue` into its section."""
        if self._key is None:
            return

        assert self._value is not None, "missing value"

        self._push(Entry(self._key, self._condition, self._value))
        self._key = self._condition = self._value = None

        self._merge_next = self._merge  # reset to default

    def key(self, token: ParsedToken) -> None:
        self.commit()
        self._key = token

    def condition(self, token: ParsedToken) -> None:
        assert self._key is not None, "condition without key"
        self._condition = token

    def value(self, token: ParsedToken) -> None:
        assert self._key is not None, "value without key"
        assert self._value is None, "duplicate value"
        self._value = token

    def open(self, token: ParsedToken) -> None:
        assert self._key is not None, "section without key"
        assert self._value is None, "open after value"
        section = Section(
            token,
            KeyValues(self._stack[-1], self._key, self._condition),
            None,
        )
        index = self._push(Entry(self._key, self._condition, section))

        # Inserted section might have been merged and discarded.
        section = self._stack[-1].children.at(index).value
        assert isinstance(section, Section)
        self._stack.append(section)

        self._key = self._condition = None

    def close(self, token: ParsedToken) -> None:
        assert len(self._stack) > 1, "close in root section"
        self.commit()
        self._stack[-1].close = token
        self._stack.pop()

    def macro(self, token: ParsedToken) -> None:
        self.key(token)
        # Don't merge macros.
        self._merge_next = False

    def token(self, token: ParsedToken) -> None:
        match token.meta.get("role"):
            case ParsedTokenRole.KEY:
                self.key(token)
            case ParsedTokenRole.CONDITION:
                self.condition(token)
            case ParsedTokenRole.VALUE:
                self.value(token)
            case ParsedTokenRole.OPEN:
                self.open(token)
            case ParsedTokenRole.CLOSE:
                self.close(token)
            case ParsedTokenRole.MACRO:
                self.macro(token)
            case None:
                if token.tag == ParsedTokenTag.EOF:
                    # Commit last `KeyValue`.
                    self.commit()
            case _ as unreachable:
                assert_never(unreachable)

    def _push(self, child: Entry[KeyValues]) -> int:
        parent = self._stack[-1].children
        return (
            parent.insert(child) if self._merge_next else parent.append(child)
        )
