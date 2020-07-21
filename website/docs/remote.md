---
id: remote
title: Remote Projects
---

:::caution Early Support for Remote Projects
The current support for remote projects is meant for *advanced users*. We are
working on a more user-friendly process for setting up a remote project that we
will ship in a future release.
:::

## Terminology
- **Client:** The local machine where you run Atom and the Skyline plugin.
- **Server:** The remote machine where you want to run the Skyline profiler.


## Prerequisites
**SSH Access.**
At a minimum, you need SSH access to a server machine that allows SSH
tunnelling. If the server machine exposes ports then it does not need to
support SSH tunnelling.

**Remote File System.**
Skyline does not provide a remote file system. As a result, to use Skyline for
a remote project, your project files must be stored in a file system that can
be accessed by both the client and server machines. Usually this is done by (i)
storing your project files in a networked file system (e.g.,
[NFS](https://en.wikipedia.org/wiki/Network_File_System)), or (ii) mounting
your project, which is stored on the server, onto your local machine using
[sshfs](https://github.com/libfuse/sshfs).


## Starting a Remote Profiling Session

### Connecting to the Server Machine
The Skyline plugin and profiler communicate over a TCP socket. As a result, we
need to ensure there is a port that they can communicate over. We recommend
using an SSH tunnel to forward ports on your client machine to the server
machine. To set up a tunnel, run:

```bash
ssh -L 60210:localhost:60210 <server hostname>
```

Skyline uses port 60210 by default, so we recommend forwarding that port. If
your server exposes ports, you do not need to set up an SSH tunnel and can use
one of the open ports instead.

### Starting the Skyline Profiler
After connecting to the server, you can start the Skyline profiler by
navigating to your project root and running the `skyline interactive` command
as usual, but with the `--skip-atom` flag added. This flag prevents Skyline
from attempting to launch Atom (since you will be running it on your client
machine).

```bash
cd ~/your/project/root
skyline interactive your_entry_point.py --skip-atom
```

If you want to use a different port, you can use the `--port` flag to tell the
profiler to listen on a different port.

```bash
skyline interactive your_entry_point.py --skip-atom --port 1337
```

### Starting the Skyline Plugin
Launch Atom and open Skyline by running the `Skyline:Toggle` command in the
command palette (Ctrl-Shift-P on Ubuntu, ⌘-Shift-P on macOS). You can also
launch Skyline using the Atom menus: Packages > Skyline > Show/Hide Skyline.

Now, *instead* of hitting Connect, click the button next to it with a gear
icon. Three text fields should appear. If you are using an SSH tunnel to
connect to the Skyline profiler, you do not need to change the host and port
(the first two text boxes). If you are connecting to a custom port on the
server machine, enter the correct host and port.

Next, you need to specify the absolute path to the project root on the client
machine. A quick way of doing this is to open a project file in Atom and then
click the button next to the text field. Skyline will fill in the project root
using the path to that file. You can then edit the path to correct it if
needed.

Once all three fields have been filled in, you can click the Connect button to
start your profiling session. If everything was set up correctly, you will be
able to use Skyline with your remote project!
