Skyline Atom Plugin
===================
This directory contains code that is specific to the Skyline Atom plugin. A
subset of these files will be copied into the plugin release repository each
time a plugin release is made.

Development Environment
-----------------------
To work on the plugin, you need:

- Atom and `apm`
- Node and `npm`

Before continuing, make sure you uninstall the Skyline plugin if you have
previously installed it through `apm` or Atom. Then:

1. Run `atom-dev-setup.sh` in this directory.
2. Run `npm ci` in this directory to install the plugin's dependencies.

The `atom-dev-setup.sh` script will create a symbolic link to the plugin code
so that you can test your changes in Atom during development without having to
reinstall the plugin on each change. If you want to uninstall the development
version of the plugin, run `./atom-dev-setup.sh --uninstall`.
