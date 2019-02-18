#! /bin/bash

SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )

UNINSTALL_OPTION="--uninstall"

function install() {
  apm link .
}

function uninstall() {
  apm unlink .
}

if [ "$1" == "$UNINSTALL_OPTION" ] ; then
  uninstall
else
  install
fi
