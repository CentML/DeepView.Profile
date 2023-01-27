**CAUTION:** Early Support for Remote Projects
The current support for remote projects is meant for *advanced users*. We are working on a more user-friendly process for setting up a remote project that we will ship in a future release.

## Terminology
- **Client:** The local machine where you run the Skyline plugin.
- **Server:** The remote machine where you want to run the Skyline profiler.

## Prerequisites
**SSH Access.**
At a minimum, you need SSH access to a server machine that allows SSH tunnelling. If the server machine exposes ports then it does not need to
support SSH tunnelling.

**Remote File System.**
Skyline does not provide a remote file system. As a result, to use Skyline for a remote project, your project files must be stored in a file system that can be accessed by both the client and server machines. Usually this is done by (i) storing your project files in a networked file system (e.g., [NFS](https://en.wikipedia.org/wiki/Network_File_System)), or (ii) mounting your project, which is stored on the server, onto your local machine using [sshfs](https://github.com/libfuse/sshfs).

## Starting a Remote Profiling Session

### Connecting to the Server Machine
The Skyline plugin and profiler communicate over a TCP socket. As a result, we need to ensure there is a port that they can communicate over. We recommend using an SSH tunnel to forward ports on your client machine to the server machine. To set up a tunnel, run:

```zsh
ssh -L 60210:localhost:60210 <server hostname>
```

Skyline uses port 60210 by default, so we recommend forwarding that port. If your server exposes ports, you do not need to set up an SSH tunnel and can use one of the open ports instead.

### Starting the Skyline Profiler
After connecting to the server, you can start the Skyline profiler by navigating to your project root and running the `skyline interactive` command as usual.

```zsh
pipenv run skyline interactive
```

If you want to use a different port, you can use the `--port` flag to tell the profiler to listen on a different port.

```zsh
pipenv skyline interactive --port 1337
```

### Starting the Skyline Plugin
Launch VSCode and open Skyline by running the `Skyline` command in the command palette (Ctrl-Shift-P on Ubuntu, ⌘-Shift-P on macOS). 