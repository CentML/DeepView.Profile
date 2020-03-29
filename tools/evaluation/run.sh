#! /bin/bash

set -e

function measure_cnn() {
  pushd ../../samples
  source ../cli/env/bin/activate

  echo_blue "Generating prediction models for $1..."
  skyline prediction-models \
    -b 8 16 32 \
    -o ../tools/evaluation/$1_models.csv \
    $1/entry_point.py

  echo_blue "Making measurements for $1..."
  skyline measure-batches \
    -b 10 20 30 40 50 60 \
    -o ../tools/evaluation/$1_measure.csv \
    -t 5 \
    $1/entry_point.py

  deactivate
  popd
}

function measure_nmt() {
  pushd ../../samples
  source ../cli/env/bin/activate

  echo_blue "Generating prediction models for $1..."
  skyline prediction-models \
    -b 32 64 80 \
    -o ../tools/evaluation/$1_models.csv \
    $1/entry_point.py

  echo_blue "Making measurements for $1..."
  skyline measure-batches \
    -b 50 70 90 110 130 150 \
    -o ../tools/evaluation/$1_measure.csv \
    -t 5 \
    $1/entry_point.py

  deactivate
  popd
}

function combine() {
  python3 process_results.py \
    --models ./$1_models.csv \
    --measurements ./$1_measure.csv \
    --output $1_combined.csv
}

function main() {
  measure_cnn "resnet"
  combine "resnet"

  measure_nmt "gnmt"
  combine "gnmt"
  measure_nmt "transformer"
  combine "transformer"

  RESULTS_DIR="results-$(date "+%F_%H_%M")"
  mkdir $RESULTS_DIR
  mkdir $RESULTS_DIR/raw
  mv *_measure.csv $RESULTS_DIR/raw
  mv *_models.csv $RESULTS_DIR/raw
  mv *.csv $RESULTS_DIR

  echo_green "Done!"
}

SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $SCRIPT_PATH
source ../shared.sh

main $@
