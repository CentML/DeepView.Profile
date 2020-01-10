'use babel';

import React from 'react';
import ReactDOM from 'react-dom';
import {Provider} from 'react-redux';

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
import AnalysisStore from './stores/analysis_store';
import ProjectStore from './stores/project_store';
import Logger from './logger';

import storeCreator from './redux/store';
import AppActions from './redux/actions/app';

// Clear the views if an analysis request is pending for more than
// this many milliseconds.
const CLEAR_VIEW_AFTER_MS = 200;

export default class SkylinePlugin {
  constructor() {
    this._store = storeCreator();
    this._handleMessage = this._handleMessage.bind(this);
    this._getStartedClicked = this._getStartedClicked.bind(this);
    this._handleServerClosure = this._handleServerClosure.bind(this);
  }

  open() {
    if (this._store.getState().appState !== AppState.ACTIVATED) {
      return;
    }
    this._store.dispatch(AppActions.appOpened());

    this._panel = atom.workspace.addRightPanel({item: document.createElement('div')});
    ReactDOM.render(
      <Provider store={this._store}>
        <PerfVis handleGetStartedClick={this._getStartedClicked} />
      </Provider>,
      this._panel.getItem(),
    );
  }

  close() {
    if (this._store.getState().appState === AppState.ACTIVATED) {
      return;
    }
    this._store.dispatch(AppActions.appClosed());
    this._disconnectFromServer();

    ReactDOM.unmountComponentAtNode(this._panel.getItem());
    this._panel.destroy();
    this._panel = null;
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
    // 1. Shutdown the connection socket
    if (this._connection != null) {
      this._connection.close();
      this._connection = null;
    }

    // 2. Discard any unneeded connection state
    this._messageHandler = null;
    this._messageSender = null;
    if (this._connectionState != null) {
      this._connectionState.dispose();
      this._connectionState = null;
    }

    Logger.info('Disconnected from the server.');
  }

  _getStartedClicked({host, port}) {
    if (INNPVStore.getAppState() !== AppState.OPENED) {
      return;
    }

    INNPVStore.setAppState(AppState.CONNECTING);
    this._connectToServer(host, port)
      .then(() => {
        this._messageSender.sendInitializeRequest();
        this._connectionState.setInitializationTimeout(() => {
          this._disconnectFromServer();
          INNPVStore.setErrorMessage(
            'Skyline timed out when establishing a connection with the Skyline server. ' +
            'Please check that the server is running.'
          );
          INNPVStore.setAppState(AppState.OPENED);
        });
      })
      .catch((err) => {
        this._connection = null;
        if (err.hasOwnProperty('errno') && err.errno === 'ECONNREFUSED') {
          INNPVStore.setErrorMessage(
            'Skyline could not connect to the Skyline server. Please check that the server ' +
            'is running and that the connection options are correct.'
          );
        } else {
          INNPVStore.setErrorMessage('Unknown error occurred. Please file a bug report!');
          Logger.error(err);
        }
        INNPVStore.setAppState(AppState.OPENED);
      });
  }

  _handleServerClosure() {
    this._disconnectFromServer();
    INNPVStore.setAppState(AppState.OPENED);
    INNPVStore.setErrorMessage(
      'Skyline has lost its connection to the server. Please check that ' +
      'the server is running before reconnecting.'
    );
    BatchSizeStore.reset();
    OperationInfoStore.reset();
    AnalysisStore.reset();
    ProjectStore.reset();
  }

  _handleMessage(message) {
    this._messageHandler.handleMessage(message);
  }
}
