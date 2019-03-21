'use babel';

import BaseStore from './base_store';
import AppState from '../models/AppState';
import PerfVisState from '../models/PerfVisState';

class INNPVStore extends BaseStore {
  constructor() {
    super();
  }

  reset() {
    this._appState = AppState.ACTIVATED;
    this._perfVisState = PerfVisState.READY;
    this._errorMessage = '';

    this._editor = null;
    this._editorOnChangeCallback = null;
    this._bufferChangedDisposable = null;
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

  setEditor(editor, onChangeCallback) {
    this.ignoreEditorChanges();
    this._editor = editor;
    this._editorOnChangeCallback = onChangeCallback;
    // NOTE: No notifyChanged() call because this doesn't affect rendering
  }

  subscribeToEditorChanges() {
    this._bufferChangedDisposable = this._editor.getBuffer().onDidChange(this._editorOnChangeCallback);
  }

  ignoreEditorChanges() {
    if (this._bufferChangedDisposable == null) {
      return;
    }
    this._bufferChangedDisposable.dispose();
    this._bufferChangedDisposable = null;
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
