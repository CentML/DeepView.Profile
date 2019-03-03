'use babel';

import EventEmitter from 'events';

import AppState from '../models/AppState';
import PerfVisState from '../models/PerfVisState';

const UPDATE_EVENT = 'updated';

class INNPVStore {
  constructor() {
    this._appState = AppState.ACTIVATED;
    this._perfVisState = PerfVisState.READY;
    this._operationInfos = [];
    this._editor = null;

    this._emitter = new EventEmitter();
  }

  getAppState() {
    return this._appState;
  }

  setAppState(state) {
    this._appState = state;
    this._notifyChanged();
  }

  getPerfVisState() {
    return this._perfVisState;
  }

  setPerfVisState(state) {
    if (this._perfVisState === state) {
      return;
    }
    this._perfVisState = state;
    this._notifyChanged();
  }

  getOperationInfos() {
    return this._operationInfos;
  }

  setOperationInfos(operationInfos) {
    this._operationInfos = operationInfos;
    this._notifyChanged();
  }

  getEditor() {
    return this._editor;
  }

  setEditor(editor) {
    this._editor = editor;
    this._notifyChanged();
  }

  addListener(callback) {
    this._emitter.on(UPDATE_EVENT, callback);
  }

  removeListener(callback) {
    this._emitter.removeListener(UPDATE_EVENT, callback);
  }

  _notifyChanged() {
    this._emitter.emit(UPDATE_EVENT);
  }
}

const storeInstance = new INNPVStore();

export default storeInstance;
