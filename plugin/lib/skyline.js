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

  handleURI(uri) {
    if (this._plugin == null) {
      this._createPlugin(
        uri.query.host,
        uri.query.port,
        uri.query.projectRoot,
      );
    }
  },

  toggle() {
    if (this._plugin == null) {
      this._createPlugin();
    } else {
      this._disposePlugin();
    }
  },

  _createPlugin(
    initialHost = null,
    initialPort = null,
    initialProjectRoot = null,
  ) {
    if (this._plugin != null) {
      return;
    }
    this._plugin = new SkylinePlugin(
      initialHost,
      initialPort,
      initialProjectRoot,
    );
  },

  _disposePlugin() {
    if (this._plugin == null) {
      return;
    }
    this._plugin.dispose();
    this._plugin = null;
  },
};
