#! /bin/bash

if [ -z "$(which python3)" ]; then
  echo "ERROR: Python 3.6+ (python3) must be installed."
  exit 1
fi

if [ -z "$(which pip3)" ]; then
  echo "ERROR: Pip for Python 3 (pip3) must be installed."
  exit 1
fi

if [ -z "$(which virtualenv)" ]; then
  echo "ERROR: Virtualenv must be installed."
  exit 1
fi

set -e
SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $SCRIPT_PATH

virtualenv -p $(which python3) env
source env/bin/activate
pip3 install \
  --global-option="--install-skyline-evaluate" \
  --editable .

echo ""
echo "Done!"
echo "A development version of Skyline was installed in the 'env' virtualenv."
echo "Activate the virtualenv by running 'source env/bin/activate' inside the 'cli' directory."
