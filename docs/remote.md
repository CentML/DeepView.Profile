# Remote Profiling

## Terminology
- **Client:** The local machine where you run the DeepView.Explore plugin.
- **Server:** The remote machine where you want to run the DeepView.Profile.

## Prerequisites
**SSH Access.**
At a minimum, you need SSH access to a server that allows SSH tunnelling. If the server machine exposes ports then it does not need to support SSH tunnelling.

**DeepView.Profile and DeepView.Predict.**
Install DeepView.Profile and (optionally DeepView.Predict) on your server to allow remote profiling.

**[VSCode Remote - SSH extension.](https://code.visualstudio.com/docs/remote/ssh)**
This extension allows users to connect to a remote machine and run extensions remotely. The extension handles most of the heavy lifting so it makes it easy to use DeepView.Explore on remote machines.

**Installing the DeepView.Explore on the Server**
To install the DeepView.Explore plugin on the server, take the following steps.
1. Connect to your server via SSH.
2. Get the VSIX file following the installation instructions. Take note the path to the VSIX file.
2. Open VSCode on your client and connect to your server.
3. Click the Extensions tab (Ctrl-Shift-X on Linux/Windows, ⌘-Shift-X on macOS) and click the `...` button. Click `Install from VSIX` and then specify the path to the VSIX file on your server.
4. Restart VSCode to enable your changes.

## Starting a Remote Profiling Session

### Starting the DeepView.Profiler
DeepView.Profile needs to running on the server to enable the plugin. You can connect to the server via SSH and start DeepView.Profile by running the `deepview interactive` command as usual.

```zsh
poetry run deepview interactive
```

If you want to use a different port, you can use the `--port` flag to tell the profiler to listen on a different port.

```zsh
poetry run deepview interactive --port portNumber
```

### Starting DeepView.Explore
Launch VSCode and open DeepView.Explore by running the deepview command in the command palette (Ctrl-Shift-P on Linux/Windows, ⌘-Shift-P on macOS). Select your project root and begin profiling.
