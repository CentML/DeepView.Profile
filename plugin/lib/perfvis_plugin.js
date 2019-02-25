'use babel';

import { CompositeDisposable } from 'atom';
import React from 'react';
import ReactDOM from 'react-dom';

import PerfVis from './components/PerfVis';
import Connection from './io/connection';
import MessageHandler from './io/message_handler';
import MessageSender from './io/message_sender';
import AppState from './models/AppState';
import INNPVStore from './stores/innpv_store';
import { getTextEditor } from './utils';

export default class PerfvisPlugin {
  constructor() {
    this._contentsChanged = this._contentsChanged.bind(this);
    this._handleMessage = this._handleMessage.bind(this);
    this._requestAnalysis = this._requestAnalysis.bind(this);
    this._getStartedClicked = this._getStartedClicked.bind(this);
  }

  open() {
    if (INNPVStore.getAppState() !== AppState.ACTIVATED) {
      return;
    }
    INNPVStore.setAppState(AppState.OPENED);

    this._panel = atom.workspace.addRightPanel({item: document.createElement('div')});
    ReactDOM.render(<PerfVis handleGetStartedClick={this._getStartedClicked} />, this._panel.getItem());

    this._subscriptions = new CompositeDisposable();
    this._connection = new Connection(this._handleMessage);
    this._messageSender = new MessageSender(this._connection);
    this._messageHandler = new MessageHandler(this._messageSender);
  }

  close() {
    if (INNPVStore.getAppState() === AppState.ACTIVATED) {
      return;
    }
    INNPVStore.setAppState(AppState.ACTIVATED);

    if (this._editorDebounce != null) {
      clearTimeout(this._editorDebounce);
      this._editorDebounce = null;
    }
    this._connection.close();
    this._connection = null;
    this._messageHandler = null;
    this._messageSender = null;
    this._editor = null;
    this._subscriptions.dispose();

    ReactDOM.unmountComponentAtNode(this._panel.getItem());
    this._panel.destroy();
    this._panel = null;
  }

  _getStartedClicked(event) {
    if (INNPVStore.getAppState() !== AppState.OPENED) {
      return;
    }

    INNPVStore.setAppState(AppState.CONNECTING);
    Promise.all([getTextEditor(), this._connection.connect('localhost', 6060)]).then((values) => {
      INNPVStore.setAppState(AppState.READY);

      this._editor = values[0];
      this._subscriptions.add(this._editor.getBuffer().onDidChange(this._contentsChanged));

      console.log('Connected!');
      this._requestAnalysis();
    });
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
}
