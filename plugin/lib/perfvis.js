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
      'perfvis:start': () => this._plugin.open(),
    }));
    this._subscriptions.add(atom.commands.add('atom-workspace', {
      'perfvis:stop': () => this._plugin.close(),
    }));
  },

  deactivate() {
    this._plugin.close();
    this._subscriptions.dispose();
  },
};
