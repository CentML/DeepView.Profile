# Remote Profiling

## Terminology
- **Client:** The local machine where you run the Skyline plugin.
- **Server:** The remote machine where you want to run the Skyline profiler.

## Prerequisites
**SSH Access.**
At a minimum, you need SSH access to a server that allows SSH tunnelling. If the server machine exposes ports then it does not need to support SSH tunnelling.

**Skyline and Habitat.**
Install Skyline and (optionally Habitat) on your server to allow remote profiling.

**[VSCode Remote - SSH extension.](https://code.visualstudio.com/docs/remote/ssh)**
This extension allows users to connect to a remote machine and run extensions remotely. The extension handles most of the heavy lifting so it makes it easy to use the Skyline plugin on remote machines.

**Installing the Skyline Plugin on the Server**
To install the Skyline plugin on the server, take the following steps.
1. Connect to your server via SSH.
2. Get the VSIX file following the installation instructions. Take note the path to the VSIX file.
2. Open VSCode on your client and connect to your server.
3. Click the Extensions tab (Ctrl-Shift-X on Linux/Windows, ⌘-Shift-X on macOS) and click the `...` button. Click `Install from VSIX` and then specify the path to the VSIX file on your server.
4. Restart VSCode to enable your changes.

## Starting a Remote Profiling Session

### Starting the Skyline Profiler
The Skyline Profiler needs to running on the server to enable the plugin. You can connect to the server via SSH and start the Skyline profiler by running the `skyline interactive` command as usual.

```zsh
poetry run skyline interactive
```

If you want to use a different port, you can use the `--port` flag to tell the profiler to listen on a different port.

```zsh
poetry run skyline interactive
```

### Starting the Skyline Plugin
Launch VSCode and open Skyline by running the Skyline command in the command palette (Ctrl-Shift-P on Linux/Windows, ⌘-Shift-P on macOS). Select your project root and begin profiling.
