#!/bin/bash

# This script is used to deploy the Skyline website.
#
# Compiled versions of the Skyline website are stored in a separate repository.
# Users need to clone a copy of the repository and pass in a path to it when
# running this tool.

set -e

function print_usage() {
  echo "Usage: $0 path/to/deploy/repository"
  echo ""
  echo "This tool is used to deploy the Skyline website."
}

function check_deploy_repo() {
  pushd "$DEPLOY_REPO"

  if [[ ! -z $(git status --porcelain) ]];
  then
    echo_red "ERROR: There are uncommitted changes in the website deploy repository. Please remove these files before deploying."
    exit 1
  fi

  # Make sure we're on master
  DEPLOY_REPO_MASTER_HASH=$(git rev-parse master)
  DEPLOY_REPO_HASH=$(git rev-parse HEAD)

  if [[ $DEPLOY_REPO_MASTER_HASH != $DEPLOY_REPO_HASH ]]; then
    echo_red "ERROR: The deploy repository must be on master when deploying."
    exit 1
  fi

  popd

  echo_green "✓ Deploy repository OK"
}

function build_website() {
  pushd ../website
  npm run build
  rm -f build/.DS_Store
  echo_green "✓ Website build succeeded"
  popd
}

function perform_deploy() {
  # Delete everything from the deploy repository
  pushd "$DEPLOY_REPO"
  rm -rf *
  popd

  # Move over the newly built website
  cp -r ../website/build/* $DEPLOY_REPO

  pushd "$DEPLOY_REPO"

  git add .
  git commit -F- <<EOF
Deploy up to commit $SHORT_HASH

This deployment includes the website files up to commit $SKYLINE_HASH in the monorepository.
EOF
  echo_green "✓ Deploy repository files updated"

  git push origin master
  echo_green "✓ Website deployed"

  popd
}

function main() {
  if [ -z "$(which node)" ]; then
    echo_red "ERROR: Node.js (node) must be installed."
    exit 1
  fi

  if [ -z "$(which npm)" ]; then
    echo_red "ERROR: npm must be installed."
    exit 1
  fi

  echo ""
  echo_blue "Skyline Website Release Tool"
  echo_blue "============================"

  echo ""
  echo_yellow "> Checking the Skyline monorepo (this repository)..."
  check_monorepo

  echo ""
  echo_yellow "> Checking the website deploy repository..."
  check_deploy_repo

  echo ""
  echo_yellow "> Tooling versions:"
  echo "npm:  $(npm -v)"
  echo "node: $(node -v)"

  SKYLINE_HASH="$(get_monorepo_hash)"
  SHORT_HASH="$(get_monorepo_short_hash)"

  echo ""
  echo_yellow "> This tool will deploy the website code at commit hash '$SKYLINE_HASH'."
  prompt_yn "> Do you want to continue? This is the final confirmation step. (y/N) "

  echo ""
  echo_yellow "> Building the website..."
  build_website

  echo ""
  echo_yellow "> Deploying the website..."
  perform_deploy

  echo_green "✓ Done!"
}

RELEASE_SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
source $RELEASE_SCRIPT_PATH/shared.sh

DEPLOY_REPO=$1
if [ -z $1 ]; then
  echo_red "ERROR: Please provide a path to the deploy repository."
  echo ""
  print_usage $@
  exit 1
fi

if [ ! -d "$DEPLOY_REPO" ]; then
  echo_red "ERROR: The deploy repository path does not exist."
  exit 1
fi

DEPLOY_REPO=$(cd "$DEPLOY_REPO" && pwd)
cd $RELEASE_SCRIPT_PATH

main $@
