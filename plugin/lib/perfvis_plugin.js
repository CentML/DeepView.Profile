'use babel';

import { CompositeDisposable } from 'atom';

export default class PerfvisPlugin {
  constructor() {
    this._isActive = false;
  }

  _initialize(editor, panel) {
    this._editor = editor;
    this._panel = panel;

    this._contentsChanged = this._contentsChanged.bind(this);

    this._subscriptions = new CompositeDisposable();
    this._subscriptions.add(this._editor.getBuffer().onDidChange(this._contentsChanged));
  }

  _contentsChanged(event) {
    console.log(event);
  }

  _getTextEditor(newEditor) {
    return new Promise((res) => {
      if (newEditor) {
        return res(atom.workspace.open());
      }
      const editor = atom.workspace.getActiveTextEditor();
      if (editor) {
        return res(editor);
      }
      // Open a new text editor if one is not open
      return res(atom.workspace.open());
    });
  }

  _getPanel() {
    const el = document.createElement('div');
    el.innerHTML = 'Hello world!';
    return atom.workspace.addRightPanel({item: el});
  }

  start() {
    if (this._isActive) {
      return;
    }
    this._isActive = true;
    this._getTextEditor().then(editor => {
      this._initialize(editor, this._getPanel());
    });
  }

  stop() {
    if (!this._isActive) {
      return;
    }
    this._isActive = false;
    this._subscriptions.dispose();
    this._panel.destroy();
    this._panel = null;
  }
}
