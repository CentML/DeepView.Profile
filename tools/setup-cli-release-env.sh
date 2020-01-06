#! /bin/bash

set -e
virtualenv -p $(which python3) renv
source renv/bin/activate
pip3 install pep517 twine

echo ""
echo "✓ Done!"
echo "NOTE: Before releasing, please ensure that your ~/.pypirc is correct."
