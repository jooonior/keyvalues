#!/usr/bin/env -S 2>/dev/null=2>NUL sh -x

mypy --no-error-summary . || exit 1
ruff check --quiet --ignore FIX . || exit 1
ruff format --diff --quiet . || exit 1
vermin --quiet --violations --eval-annotations -t=3.11 . || exit 1

exit 0

:: This is a polyglot that works both in Windows 'cmd' and Unix 'sh'.
:: The shebang however relies on flags specific to GNU 'env'.
