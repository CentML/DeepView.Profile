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
git add pyproject.toml
git commit -m "Bump version to $VERSION_TAG"
git push origin "release-$VERSION_TAG"
REPO_HASH=$(get_repo_hash)

echo ""
build_release

echo ""
echo "Release Notes:"
RELEASE_NOTES=$(git log $(git describe --abbrev=0 --tags).. --merges --pretty=format:"%s %b" | cut -f 4,7- -d ' ')

GH_TOKEN=$UOFT_ECOSYSTEM_GH_TOKEN
echo ""
echo "Create a draft release on Github"
gh release create "v$VERSION_TAG" --draft --title "v$VERSION_TAG" --notes $RELEASE_NOTES --target $REPO_HASH

case $1 in
--deploy)
echo ""
prompt_yn "> Ready to release to PyPI? (y/N) "

echo ""
echo_yellow "> Releasing $VERSION_TAG of the CLI...";;
--test-deploy)
echo ""
prompt_yn "> Ready to release to Test PyPI? (y/N) "

echo ""
echo_yellow "> Releasing $VERSION_TAG of the CLI...";;
# upload_release
*)
echo ""
echo_yellow "Skipping the upload to PyPI neither --deploy nor --test-deploy was passed."
echo_green "✓ Done!"
exit 0;;
esac

echo_green "✓ Done!"

