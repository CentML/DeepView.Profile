'use babel';

import React from 'react';
import ReactDOM from 'react-dom';

import PerfVis from './components/PerfVis';
import Connection from './io/connection';
import MessageHandler from './io/message_handler';
import MessageSender from './io/message_sender';
import ConnectionState from './io/connection_state';
import AppState from './models/AppState';
import PerfVisState from './models/PerfVisState';
import INNPVStore from './stores/innpv_store';
import BatchSizeStore from './stores/batchsize_store';
import OperationInfoStore from './stores/operationinfo_store';

// Clear the views if an analysis request is pending for more than
// this many milliseconds.
const CLEAR_VIEW_AFTER_MS = 200;

export default class PerfvisPlugin {
  constructor() {
    this._contentsChanged = this._contentsChanged.bind(this);
    this._handleMessage = this._handleMessage.bind(this);
    this._requestAnalysis = this._requestAnalysis.bind(this);
    this._getStartedClicked = this._getStartedClicked.bind(this);
    this._handleServerClosure = this._handleServerClosure.bind(this);
  }

  open() {
    if (INNPVStore.getAppState() !== AppState.ACTIVATED) {
      return;
    }
    INNPVStore.setAppState(AppState.OPENED);

    this._panel = atom.workspace.addRightPanel({item: document.createElement('div')});
    ReactDOM.render(
      <PerfVis handleGetStartedClick={this._getStartedClicked} />,
      this._panel.getItem(),
    );
  }

  close() {
    if (INNPVStore.getAppState() === AppState.ACTIVATED) {
      return;
    }
    INNPVStore.setAppState(AppState.ACTIVATED);
    this._disconnectFromServer();

    ReactDOM.unmountComponentAtNode(this._panel.getItem());
    this._panel.destroy();
    this._panel = null;

    INNPVStore.reset();
    BatchSizeStore.reset();
    OperationInfoStore.reset();
  }

  _connectToServer(host, port) {
    this._connection = new Connection(this._handleMessage, this._handleServerClosure);
    return this._connection.connect(host, port)
      .then(() => {
        this._connectionState = new ConnectionState();
        this._messageSender = new MessageSender(this._connection, this._connectionState);
        this._messageHandler = new MessageHandler(
          this._messageSender, this._connectionState);
      });
  }

  _disconnectFromServer() {
    // 1. We need to first "unbind" from the editor
    INNPVStore.ignoreEditorChanges();
    if (this._editorDebounce != null) {
      clearTimeout(this._editorDebounce);
      this._editorDebounce = null;
    }
    this._editor = null;

    // 2. Shutdown the connection socket
    if (this._connection != null) {
      this._connection.close();
      this._connection = null;
    }

    // 3. Discard any unneeded connection state
    this._messageHandler = null;
    this._messageSender = null;
    this._connectionState = null;

    console.log('Disconnected from the server.');
  }

  _getStartedClicked({host, port}) {
    if (INNPVStore.getAppState() !== AppState.OPENED) {
      return;
    }

    INNPVStore.setAppState(AppState.CONNECTING);
    this._connectToServer(host, port)
      .then(() => {
        this._messageSender.sendInitializeRequest();
      })
      .catch((err) => {
        this._connection = null;
        if (err.hasOwnProperty('errno') && err.errno === 'ECONNREFUSED') {
          INNPVStore.setErrorMessage(
            'INNPV could not connect to the INNPV server. Please check that the server ' +
            'is running and that the connection options are correct.'
          );
        } else {
          INNPVStore.setErrorMessage('Unknown error occurred. Please file a bug report!');
          console.error(err);
        }
        INNPVStore.setAppState(AppState.OPENED);
      });
  }

  _handleServerClosure() {
    this._disconnectFromServer();
    INNPVStore.setAppState(AppState.OPENED);
    INNPVStore.setErrorMessage(
      'INNPV has lost its connection to the server. Please check that ' +
      'the server is running before reconnecting.'
    );
    BatchSizeStore.reset();
    OperationInfoStore.reset();
  }

  _contentsChanged(event) {
    if (this._editorDebounce != null) {
      clearTimeout(this._editorDebounce);
    }
    this._editorDebounce = setTimeout(this._requestAnalysis, 1000);
    INNPVStore.setPerfVisState(PerfVisState.DEBOUNCING);
  }

  _handleMessage(message) {
    this._messageHandler.handleMessage(message);
  }

  _requestAnalysis() {
    console.log('Sending analysis request...');
    INNPVStore.setPerfVisState(PerfVisState.ANALYZING);
    this._messageSender.sendAnalyzeRequest(this._editor.getBuffer().getText());

    // If the request takes longer than 200 ms, we clear the view.
    // We don't clear it for fast requests to prevent screen flicker.
    OperationInfoStore.setClearViewDebounce(setTimeout(() => {
      OperationInfoStore.reset();
      OperationInfoStore.notifyChanged();
    }, CLEAR_VIEW_AFTER_MS));

    BatchSizeStore.setClearViewDebounce(setTimeout(() => {
      BatchSizeStore.reset();
      BatchSizeStore.notifyChanged();
    }, CLEAR_VIEW_AFTER_MS));
  }
}
