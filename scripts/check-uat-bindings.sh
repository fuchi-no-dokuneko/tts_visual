#!/bin/sh
set -eu

python -m behave --dry-run --no-summary features/daily_uat.feature
