'use babel';

import ua from 'universal-analytics';

import env from '../env';
import Logger from '../logger';

export default class TelemetryClient {
  constructor({uaId}) {
    this._visitor = ua(uaId);
  }

  record(eventType) {
    if (env.development) {
      Logger.debug('Event:', eventType.name);
      return;
    }
    this._visitor.event(eventType.category, eventType.action).send();
  }
};
