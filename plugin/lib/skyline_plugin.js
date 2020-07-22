'use babel';

import React from 'react';
import ReactDOM from 'react-dom';
import {Provider} from 'react-redux';

import PerfVis from './components/PerfVis';
import AppState from './models/AppState';
import SkylineSession from './skyline_session';
import Logger from './logger';
import env from './env.json';
import ConfigManager from './config/manager';

import Events from './telemetry/events';
import TelemetryClientContext from './telemetry/react_context';
import TelemetryClient from './telemetry/client';

import AppActions from './redux/actions/app';
import ConnectionActions from './redux/actions/connection';
import storeCreator from './redux/store';
import ConnectionStateView from './redux/views/connection_state';

export default class SkylinePlugin {
  constructor(initialHost = null, initialPort = null, initialProjectRoot = null) {
    this._store = storeCreator();
    this._session = null;
    this._telemetryClient = TelemetryClient.from(env.uaId, this._store);
    this._configManager = new ConfigManager(this._store);

    this._initialHost = initialHost;
    this._initialPort = initialPort;
    this._initialProjectRoot = initialProjectRoot;

    this._getStartedClicked = this._getStartedClicked.bind(this);
    this._triggerProfiling = this._triggerProfiling.bind(this);
    this._handleServerClosure = this._handleServerClosure.bind(this);
    this._handleInitializationTimeout = this._handleInitializationTimeout.bind(this);
    this._disposeSession = this._disposeSession.bind(this);
    this._disposeSessionAsync = () => {
      setTimeout(this._disposeSession);
    }

    this._activate();
  }

  _activate() {
    if (this._store.getState().appState !== AppState.ACTIVATED) {
      return;
    }
    this._store.dispatch(AppActions.appOpened());
    this._telemetryClient.record(Events.Skyline.OPENED);

    this._panel = atom.workspace.addRightPanel({item: document.createElement('div')});
    ReactDOM.render(
      <Provider store={this._store}>
        <TelemetryClientContext.Provider value={this._telemetryClient}>
          <PerfVis
            handleGetStartedClick={this._getStartedClicked}
            triggerProfiling={this._triggerProfiling}
            initialHost={this._initialHost}
            initialPort={this._initialPort}
            initialProjectRoot={this._initialProjectRoot}
          />
        </TelemetryClientContext.Provider>
      </Provider>,
      this._panel.getItem(),
    );
  }

  dispose() {
    if (this._store.getState().appState === AppState.ACTIVATED) {
      return;
    }
    this._store.dispatch(AppActions.appClosed());
    this._disposeSession();

    ReactDOM.unmountComponentAtNode(this._panel.getItem());
    this._panel.destroy();
    this._panel = null;

    this._configManager.dispose();
    this._configManager = null;
    this._telemetryClient = null;

    // We purposely do not set the store to null since we use
    // it to check the application state in this method.
  }

  _disposeSession() {
    if (this._session == null) {
      return;
    }
    this._session.dispose();
    this._session = null;
    Logger.info('Disconnected from the server.');
  }

  _getStartedClicked({host, port, projectRoot}) {
    if (this._store.getState().appState !== AppState.OPENED) {
      return;
    }

    this._store.dispatch(ConnectionActions.connecting());
    this._session = new SkylineSession({
      store: this._store,
      telemetryClient: this._telemetryClient,
      handleServerClosure: this._handleServerClosure,
      handleInitializationTimeout: this._handleInitializationTimeout,
      projectRoot,
      disposeSessionAsync: this._disposeSessionAsync,
    });
    this._session.connect(host, port)
      .catch((err) => {
        let errorMessage = null;
        if (err.hasOwnProperty('errno') && err.errno === 'ECONNREFUSED') {
          errorMessage =
            'Skyline could not connect to the Skyline server. Please check that the server ' +
            'is running and that the connection options are correct.';
        } else {
          errorMessage = 'Unknown error occurred. Please file a bug report!';
          Logger.error(err);
        }

        this._store.dispatch(ConnectionActions.error({errorMessage}));
        this._telemetryClient.record(Events.Error.CONNECTION_ERROR);
        this._disposeSession();
      });
  }

  _triggerProfiling() {
    if (this._session == null) {
      return;
    }
    this._session.triggerProfiling();
  }

  _handleServerClosure() {
    this._store.dispatch(ConnectionActions.lost({
      errorMessage: 'Skyline has lost its connection to the server. ' +
        'Please check that the server is running before reconnecting.',
    }));
    this._disposeSession();
  }

  _handleInitializationTimeout() {
    this._store.dispatch(ConnectionActions.error({
      errorMessage: 'Skyline timed out when establishing a connection ' +
        'with the Skyline server. Please check that the server is running.',
    }));
    this._telemetryClient.record(Events.Error.CONNECTION_TIMEOUT);
    this._disposeSession();
  }
}
