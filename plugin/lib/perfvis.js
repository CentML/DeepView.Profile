'use babel';

import { CompositeDisposable } from 'atom';
import PerfvisPlugin from './perfvis_plugin';

export default {
  _plugin: null,
  _subscriptions: null,

  activate() {
    this._plugin = new PerfvisPlugin();

    this._subscriptions = new CompositeDisposable();
    this._subscriptions.add(atom.commands.add('atom-workspace', {
      'perfvis:start': () => this._start(),
    }));
    this._subscriptions.add(atom.commands.add('atom-workspace', {
      'perfvis:stop': () => this._stop(),
    }));
  },

  deactivate() {
    this._stop();
    this._subscriptions.dispose();
  },

  _start() {
    this._plugin.start();
  },

  _stop() {
    this._plugin.stop();
  },
};
