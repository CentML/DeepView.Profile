#!/bin/bash

# This script is used to release a new version of the Skyline Atom Plugin.
#
# Released versions of the Skyline Atom Plugin are stored in a separate release
# repository. Users need to clone a copy of the release repository and pass in
# a path to it when running this tool.

set -e

function print_usage() {
  echo "Usage: $0 path/to/release/repository"
  echo ""
  echo "This tool is used to release a new version of the Skyline Atom Plugin."
}

function check_release_repo() {
  pushd "$RELEASE_REPO"

  if [[ ! -z $(git status --porcelain) ]];
  then
    echo_red "ERROR: There are uncommitted changes in the release repository. Please remove these files before releasing."
    exit 1
  fi

  # Make sure we're on master
  RELEASE_REPO_MASTER_HASH=$(git rev-parse master)
  RELEASE_REPO_HASH=$(git rev-parse HEAD)

  if [[ $RELEASE_REPO_MASTER_HASH != $RELEASE_REPO_HASH ]]; then
    echo_red "ERROR: The release repository must be on master when releasing."
    exit 1
  fi

  popd

  echo_green "✓ Release repository OK"
}

function perform_release() {
  # Delete everything from the release repository
  pushd "$RELEASE_REPO"
  rm -rf lib styles
  rm -f package.json package-lock.json .gitignore
  popd

  # Move over new copies of the plugin files
  cp -r ../plugin/lib $RELEASE_REPO
  cp -r ../plugin/styles $RELEASE_REPO
  cp ../plugin/package.json $RELEASE_REPO
  cp ../plugin/package-lock.json $RELEASE_REPO
  cp ../plugin/.gitignore $RELEASE_REPO

  pushd "$RELEASE_REPO"

  # This is how we use different log levels for development and production
  rm lib/logger.js
  mv lib/logger_prod.js lib/logger.js

  git add .
  git commit -F- <<EOF
[$VERSION_TAG] Release up to commit $SHORT_HASH

This release includes the plugin files up to commit $SKYLINE_HASH in the monorepository.
EOF
  echo_green "✓ Release repository files updated"

  git tag -a "$VERSION_TAG" -m ""
  git push --follow-tags
  echo_green "✓ Release pushed to GitHub"

  apm publish --tag "$VERSION_TAG"
  echo "✓ Release published to the Atom package index"
  popd
}

function main() {
  if [ -z "$(which apm)" ]; then
    echo_red "ERROR: The Atom package manager (apm) must be installed."
    exit 1
  fi

  if [ -z "$(which node)" ]; then
    echo_red "ERROR: Node.js (node) must be installed."
    exit 1
  fi

  echo ""
  echo_blue "Skyline Atom Plugin Release Tool"
  echo_blue "================================"

  echo ""
  echo_yellow "> Checking the Skyline monorepo (this repository)..."
  check_monorepo

  echo ""
  echo_yellow "> Checking the plugin release repository..."
  check_release_repo

  echo ""
  echo_yellow "> Tooling versions:"
  apm -v

  NEXT_PLUGIN_VERSION=$(node -p "require('../plugin/package.json').version")
  VERSION_TAG="v$NEXT_PLUGIN_VERSION"
  SKYLINE_HASH="$(get_monorepo_hash)"
  SHORT_HASH="$(get_monorepo_short_hash)"

  echo ""
  echo_yellow "> The next plugin version will be '$VERSION_TAG'."
  prompt_yn "> Is this correct? (y/N) "

  echo ""
  echo_yellow "> This tool will release the plugin code at commit hash '$SKYLINE_HASH'."
  prompt_yn "> Do you want to continue? This is the final confirmation step. (y/N) "

  echo ""
  echo_yellow "> Releasing $VERSION_TAG of the plugin..."
  perform_release

  echo_green "✓ Done!"
}

RELEASE_SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
source $RELEASE_SCRIPT_PATH/shared.sh

RELEASE_REPO=$1
if [ -z $1 ]; then
  echo_red "ERROR: Please provide a path to the release repository."
  echo ""
  print_usage $@
  exit 1
fi

if [ ! -d "$RELEASE_REPO" ]; then
  echo_red "ERROR: The release repository path does not exist."
  exit 1
fi

RELEASE_REPO=$(cd "$RELEASE_REPO" && pwd)
cd $RELEASE_SCRIPT_PATH

main $@
