'use babel';

import { CompositeDisposable } from 'atom';
import React from 'react';
import ReactDOM from 'react-dom';

import PerfVis from './components/PerfVis';
import Connection from './io/connection';
import MessageHandler from './io/message_handler';
import MessageSender from './io/message_sender';

export default class PerfvisPlugin {
  constructor() {
    this._isActive = false;
  }

  _initialize(editor) {
    this._editor = editor;
    this._panel = atom.workspace.addRightPanel({item: document.createElement('div')});
    ReactDOM.render(<PerfVis/>, this._panel.getItem());

    this._contentsChanged = this._contentsChanged.bind(this);
    this._handleMessage = this._handleMessage.bind(this);
    this._requestAnalysis = this._requestAnalysis.bind(this);

    this._subscriptions = new CompositeDisposable();
    this._subscriptions.add(this._editor.getBuffer().onDidChange(this._contentsChanged));

    this._connection = new Connection(this._handleMessage);
    this._messageSender = new MessageSender(this._connection);
    this._messageHandler = new MessageHandler(this._messageSender);

    this._connection.connect(() => {
      console.log('Connected!');
      this._requestAnalysis();
    });

    this._editorDebounce = null;
  }

  _contentsChanged(event) {
    if (this._editorDebounce != null) {
      clearTimeout(this._editorDebounce);
    }
    this._editorDebounce = setTimeout(this._requestAnalysis, 1000);
  }

  _handleMessage(message) {
    this._messageHandler.handleMessage(message);
  }

  _requestAnalysis() {
    console.log('Sending analysis request...');
    this._messageSender.sendAnalyzeRequest(this._editor.getBuffer().getText());
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

  start() {
    if (this._isActive) {
      return;
    }
    this._isActive = true;
    this._getTextEditor().then(editor => {
      this._initialize(editor);
    });
  }

  stop() {
    if (!this._isActive) {
      return;
    }

    if (this._editorDebounce != null) {
      clearTimeout(this._editorDebounce);
      this._editorDebounce = null;
    }
    this._isActive = false;
    this._connection.close();
    this._messageHandler = null;
    this._messageSender = null;
    this._subscriptions.dispose();

    ReactDOM.unmountComponentAtNode(this._panel.getItem());
    this._panel.destroy();
    this._panel = null;
  }
}
