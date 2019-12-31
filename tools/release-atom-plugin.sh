#!/bin/bash

# This script is used to release a new version of the INNPV Atom Plugin.
#
# Released versions of the INNPV Atom Plugin are stored in a separate release
# repository. Users need to clone a copy of the release repository and pass in
# a path to it when running this tool.

set -e

function print_usage() {
  echo "Usage: $0 path/to/release/repository"
  echo ""
  echo "This tool is used to release a new version of the INNPV Atom Plugin."
}

function prompt_yn() {
  read -p "$1" -r
  if [[ ! $REPLY =~ ^[Yy]$ ]]
  then
    exit 1
  fi
}

function pushd() {
    command pushd "$@" > /dev/null
}

function popd() {
    command popd "$@" > /dev/null
}

function check_monorepo() {
  # Make sure everything has been committed
  if [[ ! -z $(git status --porcelain) ]];
  then
    echo "ERROR: There are uncommitted changes. Please commit before releasing."
    exit 1
  fi

  # Make sure we're on master
  INNPV_MASTER_HASH=$(git rev-parse master)
  INNPV_HASH=$(git rev-parse HEAD)

  if [[ $INNPV_MASTER_HASH != $INNPV_HASH ]]; then
    echo "ERROR: You must be on master when releasing."
    exit 1
  fi

  INNPV_SHORT_HASH=$(git rev-parse --short HEAD)

  echo "✓ Repository OK"
}

function check_release_repo() {
  pushd "$RELEASE_REPO"

  if [[ ! -z $(git status --porcelain) ]];
  then
    echo "ERROR: There are uncommitted changes in the release repository. Please remove these files before releasing."
    exit 1
  fi

  # Make sure we're on master
  RELEASE_REPO_MASTER_HASH=$( git rev-parse master )
  RELEASE_REPO_HASH=$( git rev-parse HEAD )

  if [[ $RELEASE_REPO_MASTER_HASH != $RELEASE_REPO_HASH ]]; then
    echo "ERROR: The release repository must be on master when releasing."
    exit 1
  fi

  popd

  echo "✓ Release repository OK"
}

function perform_release() {
  # Delete everything from the release repository
  pushd "$RELEASE_REPO"
  rm -rf lib styles
  rm -f package.json package-lock.json README.md .gitignore
  popd

  # Move over new copies of the plugin files
  cp -r ../plugin/lib $RELEASE_REPO
  cp -r ../plugin/styles $RELEASE_REPO
  cp ../plugin/package.json $RELEASE_REPO
  cp ../plugin/package-lock.json $RELEASE_REPO
  cp ../plugin/README.md $RELEASE_REPO
  cp ../plugin/.gitignore $RELEASE_REPO

  pushd "$RELEASE_REPO"
  git add .
  git commit -F- <<EOF
[$VERSION_TAG] Release up to commit $INNPV_SHORT_HASH

This release includes the plugin files up to commit $INNPV_HASH in the monorepository.
EOF
  echo "✓ Release repository files updated"

  git tag -a "$VERSION_TAG" -m ""
  git push --follow-tags
  echo "✓ Release pushed to GitHub"

  # apm publish --tag $RELEASE_TAG
  # echo "✓ Release published to the Atom package index"
  popd
}

function main() {
  if [ -z "$(which apm)" ]; then
    echo "ERROR: The Atom package manager (apm) must be installed."
    exit 1
  fi

  if [ -z "$(which node)" ]; then
    echo "ERROR: Node.js (node) must be installed."
    exit 1
  fi

  echo "INNPV Atom Plugin Release Tool"
  echo "=============================="

  echo ""
  echo "> Checking the INNPV monorepo (this repository)..."
  check_monorepo

  echo ""
  echo "> Checking the plugin release repository..."
  check_release_repo

  echo ""
  echo "> Tooling versions:"
  apm -v

  NEXT_PLUGIN_VERSION=$(node -p "require('../plugin/package.json').version")
  VERSION_TAG="v$NEXT_PLUGIN_VERSION"

  echo ""
  echo "> The next plugin version will be '$VERSION_TAG'."
  prompt_yn "> Is this correct? (y/N) "

  echo ""
  echo "> This tool will release the plugin code at commit hash '$INNPV_HASH'."
  prompt_yn "> Do you want to continue? This is the final confirmation step. (y/N) "

  echo ""
  echo "> Releasing $VERSION_TAG of the plugin..."
  perform_release

  echo "✓ Done!"
}

RELEASE_REPO=$1
if [ -z $1 ]; then
  echo "ERROR: Please provide a path to the release repository."
  echo ""
  print_usage $@
  exit 1
fi

if [ ! -d "$RELEASE_REPO" ]; then
  echo "ERROR: The release repository path does not exist."
  exit 1
fi

RELEASE_REPO=$(cd "$RELEASE_REPO" && pwd)
RELEASE_SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $RELEASE_SCRIPT_PATH

main $@
