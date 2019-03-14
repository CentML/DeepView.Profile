'use babel';

import BaseStore from './base_store';
import AppState from '../models/AppState';
import PerfVisState from '../models/PerfVisState';

class INNPVStore extends BaseStore {
  constructor() {
    super();
    this._appState = AppState.ACTIVATED;
    this._perfVisState = PerfVisState.READY;
    this._errorMessage = '';
    this._editor = null;
  }

  getAppState() {
    return this._appState;
  }

  setAppState(state) {
    this._appState = state;
    this.notifyChanged();
  }

  getPerfVisState() {
    return this._perfVisState;
  }

  setPerfVisState(state) {
    if (this._perfVisState === state) {
      return;
    }
    this._perfVisState = state;
    this.notifyChanged();
  }

  getEditor() {
    return this._editor;
  }

  setEditor(editor) {
    this._editor = editor;
    // NOTE: No notifyChanged() call because this doesn't affect rendering
  }

  getErrorMessage() {
    return this._errorMessage;
  }

  setErrorMessage(message) {
    if (this._errorMessage === message) {
      return;
    }
    this._errorMessage = message;
    this.notifyChanged();
  }

  clearErrorMessage() {
    this.setErrorMessage('');
  }
}

const storeInstance = new INNPVStore();

export default storeInstance;
