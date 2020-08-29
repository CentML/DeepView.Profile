---
id: settings
title: Settings
---

This page describes the settings available in the Skyline interactive profiler
(i.e. the Skyline Atom plugin). You can access Skyline's settings through
Atom's preferences pane just like any other Atom package (Atom >
Settings/Preferences > Packages > Skyline > Settings).


### Disable Profile on Save

By default, Skyline will profile your model whenever you change and then save
your code. You can use this setting to prevent Skyline from doing this
automatic profiling.

If you do disable profiling on save, Skyline will display a message in its
status bar (bottom right corner of the Atom window) when its performance
visualizations could be out of date. To request re-profiling when this happens,
click the "cycle" button that will appear on the bottom right corner of the
Atom window.

:::note
Skyline's profiling takes place in the background and does not interfere with
your ability to write code and use Atom. As a result, we recommend *not*
disabling profiling on save.
:::


### Enable Usage Statistics Reporting

Skyline will collect anonymous usage statistics to help us understand how
Skyline is being used and to help us make further improvements to Skyline. If
you do not want Skyline to send these usage statistics, you can use this
setting to disable their collection.
