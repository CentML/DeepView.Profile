'use babel';

import {CompositeDisposable} from 'atom';
import configSchema from './schema';
import Logger from '../logger';

import ConfigActions from '../redux/actions/config';

// The purpose of this class is to subscribe to our configuration schemas
// through Atom's configuration API and to update our store with the latest
// values. This helps reduce the coupling between our components and the
// Atom APIs.
export default class ConfigManager {
  constructor(store) {
    this._store = store;
    this._subscriptions = this._subscribeToConfigs();
  }

  dispose() {
    this._subscriptions.dispose();
    this._store = null;
  }

  _subscribeToConfigs() {
    const subscriptions = new CompositeDisposable();
    for (const key in configSchema) {
      if (!Object.prototype.hasOwnProperty.call(configSchema, key)) {
        continue;
      }
      subscriptions.add(atom.config.observe(
        `skyline.${key}`, this._handleConfigChange.bind(this, key)));
    }
    return subscriptions;
  }

  _handleConfigChange(key, value) {
    if (this._store == null) {
      return;
    }
    this._store.dispatch(ConfigActions.configChanged({
      [key]: value,
    }));
  }
};
