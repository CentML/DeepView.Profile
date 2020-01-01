function pushd() {
    command pushd "$@" > /dev/null
}

function popd() {
    command popd "$@" > /dev/null
}

COLOR_RED="\033[0;31m"
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[0;33m"
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

function prompt_yn() {
  echo -en "${COLOR_YELLOW}$1${COLOR_NC}"
  read -r
  if [[ ! $REPLY =~ ^[Yy]$ ]]
  then
    exit 1
  fi
}

function get_monorepo_hash() {
  echo "$(git rev-parse HEAD)"
}

function get_monorepo_short_hash() {
  echo "$(git rev-parse --short HEAD)"
}

function check_monorepo() {
  # Make sure everything has been committed
  if [[ ! -z $(git status --porcelain) ]];
  then
    echo_red "ERROR: There are uncommitted changes. Please commit before releasing."
    exit 1
  fi

  # Make sure we're on master
  INNPV_MASTER_HASH=$(git rev-parse master)
  INNPV_HASH=$(git rev-parse HEAD)

  if [[ $INNPV_MASTER_HASH != $INNPV_HASH ]]; then
    echo_red "ERROR: You must be on master when releasing."
    exit 1
  fi

  INNPV_SHORT_HASH=$(git rev-parse --short HEAD)

  echo_green "✓ Repository OK"
}
