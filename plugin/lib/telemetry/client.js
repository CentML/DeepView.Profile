'use babel';

import ua from 'universal-analytics';
import uuidv4 from 'uuid/v4';

import env from '../env';
import Logger from '../logger';

const CLIENT_ID_KEY = 'skyline:telemetry:clientId';

export default class TelemetryClient {
  constructor({uaId, clientId, store}) {
    if (uaId == null) {
      // If a universal analytics ID is not provided, or if
      // the plugin is in development mode, this client will
      // just log the events to the console.
      this._visitor = null;
    } else {
      this._visitor = ua(uaId, clientId);
    }
    this._store = store;
  }

  static from(uaId, store) {
    let clientId = localStorage.getItem(CLIENT_ID_KEY);
    if (clientId == null) {
      clientId = uuidv4();
      localStorage.setItem(CLIENT_ID_KEY, clientId);
    }
    return new TelemetryClient({uaId, clientId, store});
  }

  record(eventType, {label, value} = {}) {
    if (eventType == null) {
      Logger.warn('Attempted to record an undefined event type.');
      return;
    }

    if (!(this._store.getState().config.enableTelemetry)) {
      // We do not send events if the user has disabled them.
      return;
    }

    if (env.development || this._visitor == null) {
      // During development, or if a universal analytics ID was not supplied, we
      // will log the event instead of sending it.
      Logger.debug('Event:', eventType.name, '| Label:', label, '| Value:', value);
      return;
    }

    this._visitor.event(
      eventType.category, eventType.action, label, value).send();
  }
};
