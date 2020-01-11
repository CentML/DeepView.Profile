'use babel';

import ConnectionActions from '../actions/connection';

export default class ConnectionStateView {
  constructor(store) {
    this._store = store;
  }

  get _connectionState() {
    return this._store.getState().connection;
  }

  isInitialized() {
    return this._connectionState.initialized;
  }

  nextSequenceNumber() {
    const nextSequenceNumber = this._connectionState.sequenceNumber;
    this._store.dispatch(ConnectionActions.incrementSequence());
    return nextSequenceNumber;
  }

  isResponseCurrent(responseSequenceNumber) {
    // Since we always increase the sequence number by one, a "current"
    // response is one with a sequence number exactly one less than the next
    // sequence number to be assigned (this._sequenceNumber).
    return responseSequenceNumber === this._connectionState.sequenceNumber - 1;
  }
}
