'use babel';

import { CompositeDisposable } from 'atom';
import INNPVPlugin from './innpv_plugin';

export default {
  _plugin: null,
  _subscriptions: null,

  activate() {
    this._plugin = new INNPVPlugin();

    this._subscriptions = new CompositeDisposable();
    this._subscriptions.add(atom.commands.add('atom-workspace', {
      'innpv:start': () => this._plugin.open(),
    }));
    this._subscriptions.add(atom.commands.add('atom-workspace', {
      'innpv:stop': () => this._plugin.close(),
    }));
  },

  deactivate() {
    this._plugin.close();
    this._subscriptions.dispose();
  },
};
