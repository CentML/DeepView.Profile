#!/bin/bash

# This script is used to release a new version of the Skyline CLI.
# Release steps:
# 1. Create release branch
# 2. Increment package version in pyproject.toml
# 3. Prepare change log since the last version 
# 4. Commit the change log
# 5. Creater draft Github release
# 6. Optional: create ability to publish to test pypi


set -e

RELEASE_SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $RELEASE_SCRIPT_PATH
source common.sh

echo ""
echo_blue "Skyline Profiler Release Preparation Tool"
echo_blue "========================================="

echo ""
check_repo

echo ""
check_tools

CURR_CLI_VERSION=$(poetry version --short)
echo -en "${COLOR_YELLOW}Release increment: [patch], minor, major${COLOR_NC}"
read -r
case $REPLY in 
major)
  poetry version major;;
minor)
  poetry version minor;;
*)
  poetry version patch;;
esac
NEXT_CLI_VERSION=$(poetry version --short)
VERSION_TAG="v$NEXT_CLI_VERSION"

echo ""
echo_yellow "> The next CLI version will be '$VERSION_TAG'."
prompt_yn "> Is this correct? (y/N) "
git checkout -b "release-$VERSION_TAG"
git commit -am "Bump version to $VERSION_TAG"
git push origin "release-$VERSION_TAG"
REPO_HASH=$(get_repo_hash)

echo ""
build_release

RELEASE_NOTES=$(git log $(git describe --abbrev=0 --tags).. --merges --pretty=format:"%s %b" | cut -f 4,7- -d ' ')
echo ""
echo "Release Notes:"
echo "$RELEASE_NOTES"

RELEASE_ARTIFACTS=$(find ../dist -name "*$NEXT_CLI_VERSION*" -type f | paste -s -d ' ' - )

GH_TOKEN=$UOFT_ECOSYSTEM_GH_TOKEN
echo ""
prompt_yn "> Create a draft release on Github? (y/N) "
gh release create "v$VERSION_TAG" --draft \
                                  --title "$VERSION_TAG" \
                                  --notes "$RELEASE_NOTES" \
                                  --target "$REPO_HASH" \
                                  $RELEASE_ARTIFACTS
echo -en "${COLOR_YELLOW}Ready to publish? [dryrun], test-pypi, pypi${COLOR_NC}"
read -r
case $REPLY in 
test-pypi)
  echo ""
  echo_yellow "> Releasing $VERSION_TAG of the CLI...";;
pypi)
  echo ""
  echo_yellow "> Releasing $VERSION_TAG of the CLI...";;
*)
  echo ""
  echo_yellow "Skipping the upload to PyPI";;
esac

echo_green "✓ Done!"

