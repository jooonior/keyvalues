#!/usr/bin/env -S 2>/dev/null=2>NUL sh -x

: : This is a polyglot that works both in Windows 'cmd' and Unix 'sh'. The shebang however relies on flags specific to GNU 'env'.

mypy --no-error-summary . || exit 1
ruff check --quiet . || exit 1
ruff format --diff --quiet . || exit 1
