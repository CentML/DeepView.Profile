'use babel';

import { CompositeDisposable } from 'atom';
import SkylinePlugin from './skyline_plugin';

export default {
  _plugin: null,
  _subscriptions: null,

  activate() {
    this._plugin = new SkylinePlugin();

    this._subscriptions = new CompositeDisposable();
    this._subscriptions.add(atom.commands.add('atom-workspace', {
      'skyline:open': () => this._plugin.open(),
    }));
    this._subscriptions.add(atom.commands.add('atom-workspace', {
      'skyline:close': () => this._plugin.close(),
    }));
  },

  deactivate() {
    this._plugin.close();
    this._subscriptions.dispose();
  },
};
