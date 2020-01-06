#! /bin/bash

SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )
cd $SCRIPTPATH

UNINSTALL_OPTION="--uninstall"
HELP_OPTION="--help"

function install() {
  apm link .
}

function uninstall() {
  apm unlink .
}

if [ "$1" == "$HELP_OPTION" ]; then
  echo "Usage: $0 [--uninstall]"
  echo ""
  echo "This script installs (or uninstalls) a development version of the "
  echo "Skyline Atom plugin."
  echo ""
  echo "Use the --uninstall flag to uninstall the development version of the "
  echo "plugin."

elif [ "$1" == "$UNINSTALL_OPTION" ]; then
  uninstall
else
  install
fi
