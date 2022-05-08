COLOR_RED="\033[0;31m"
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[0;33m"
COLOR_BLUE="\033[0;36m"
COLOR_NC="\033[0m"

function echo_colored() {
  echo -e "${1}${2}${COLOR_NC}"
}

function echo_green() {
  echo_colored "$COLOR_GREEN" "$1"
}

function echo_red() {
  echo_colored "$COLOR_RED" "$1"
}

function echo_yellow() {
  echo_colored "$COLOR_YELLOW" "$1"
}

function echo_blue() {
  echo_colored "$COLOR_BLUE" "$1"
}

function prompt_yn() {
  echo -en "${COLOR_YELLOW}$1${COLOR_NC}"
  read -r
  if [[ ! $REPLY =~ ^[Yy]$ ]]
  then
    exit 1
  fi
}

function get_repo_hash() {
  echo "$(git rev-parse HEAD)"
}

function check_repo() {
  # Make sure no unstaged changes
  echo_yellow "> Check for uncommitted changes"
  if [[ ! -z $(git status --porcelain) ]];
  then
    echo_red "ERROR: There are uncommitted changes. Please commit before releasing."
    exit 1
  fi

  # Make sure we're on main
  echo_yellow "> Check the current branch"
  INNPV_MAIN_HASH=$(git rev-parse main)
  INNPV_HASH=$(git rev-parse HEAD)

  if [[ $INNPV_MAIN_HASH != $INNPV_HASH ]]; then
    echo_red "ERROR: You must be on main when releasing."
    exit 1
  fi

  INNPV_SHORT_HASH=$(git rev-parse --short HEAD)

  echo_green "✓ Repository OK"
}

function check_tools() {
  echo_yellow "> Check tools"
  if [ -z "$(which poetry)" ]; then
    echo_red "ERROR: Poetry must be installed."
    exit 1
  fi

  if [ -z "$(which gh)" ]; then
    echo_red "ERROR: GitHub CLI must be installed."
    exit 1
  fi

  echo ""
  echo_yellow "> Tooling versions:"
  echo "$(poetry --version)"
  echo "$(poetry run python3 --version)"
  echo "$(gh --version)"
  echo_green "✓ Release tooling OK"
}

function build_release() {
  echo_yellow "> Building wheels..."
  rm -rf ../dist/*
  cp ../pyproject.toml ../skyline/
  poetry build
  echo_green "✓ Wheels successfully built"
}

function upload_release() {
  echo ""
  echo_yellow "> Uploading release to PyPI..."
  twine upload -r pypi "dist/skyline_cli-${NEXT_CLI_VERSION}*"
  echo_green "✓ New release uploaded to PyPI"

  echo ""
  echo_yellow "> Creating a release tag..."
  git tag -a "$VERSION_TAG" -m ""
  git push --follow-tags
  echo_green "✓ Git release tag created and pushed to GitHub"
}

