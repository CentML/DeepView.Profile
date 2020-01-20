'use babel';

import ua from 'universal-analytics';

import env from '../env';
import Logger from '../logger';

export default class TelemetryClient {
  constructor({uaId}) {
    if (uaId == null) {
      // If a universal analytics ID is not provided, or if
      // the plugin is in development mode, this client will
      // just log the events to the console.
      this._visitor = null;
    } else {
      this._visitor = ua(uaId);
    }
  }

  record(eventType, {label, value} = {}) {
    if (eventType == null) {
      Logger.warn('Attempted to record an undefined event type.');
      return;
    }

    if (env.development || this._visitor == null) {
      Logger.debug('Event:', eventType.name, '| Label:', label, '| Value:', value);
      return;
    }

    this._visitor.event(
      eventType.category, eventType.action, label, value).send();
  }
};
