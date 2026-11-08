#!/usr/bin/env bash
# deploy webpage/ to GitHub Pages via the gh-pages branch
set -euo pipefail
cd "$(dirname "$0")"
git subtree push --prefix=webpage origin gh-pages
