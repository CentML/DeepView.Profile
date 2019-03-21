'use babel';

import EventEmitter from 'events';

const UPDATE_EVENT = 'updated';

export default class BaseStore {
  constructor() {
    this._emitter = new EventEmitter();
    this.reset();
  }

  reset() {
    // Implemented by child classes. Used to initialize/reset the store's state.
  }

  addListener(callback) {
    this._emitter.on(UPDATE_EVENT, callback);
  }

  removeListener(callback) {
    this._emitter.removeListener(UPDATE_EVENT, callback);
  }

  notifyChanged() {
    this._emitter.emit(UPDATE_EVENT);
  }
}
