#! /bin/bash

set -e

function measure() {
  pushd ../../samples
  source ../cli/env/bin/activate

  echo_blue "Generating prediction models..."
  skyline prediction-models \
    -b 8 16 32 \
    -o ../tools/evaluation/models.csv \
    resnet/entry_point.py

  echo_blue "Making measurements..."
  skyline measure-batches \
    -b 10 20 30 40 50 \
    -o ../tools/evaluation/measure.csv \
    resnet/entry_point.py

  popd
}

function main() {
  measure

  RESULTS_DIR="results-$(date "+%F_%H_%M")"
  mkdir $RESULTS_DIR
  mv *.csv $RESULTS_DIR

  echo_green "Done!"
}

SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $SCRIPT_PATH
source ../shared.sh

main $@
