'use babel';

import {CompositeDisposable, Disposable} from 'atom';
import SkylinePlugin from './skyline_plugin';
import configSchema from './config/schema';

export default {
  _plugin: null,
  _subscriptions: null,

  activate() {
    this._subscriptions = new CompositeDisposable(
      atom.commands.add('atom-workspace', {
        'skyline:toggle': () => this.toggle(),
      }),
      new Disposable(this._disposePlugin.bind(this)),
    );
  },

  deactivate() {
    this._subscriptions.dispose();
  },

  config: configSchema,

  toggle() {
    if (this._plugin == null) {
      this._plugin = new SkylinePlugin();
    } else {
      this._disposePlugin();
    }
  },

  _disposePlugin() {
    if (this._plugin == null) {
      return;
    }
    this._plugin.dispose();
    this._plugin = null;
  },
};
