'use babel';

import React from 'react';
import ReactDOM from 'react-dom';
import {Provider} from 'react-redux';

import PerfVis from './components/PerfVis';
import AppState from './models/AppState';
import SkylineSession from './skyline_session';
import Logger from './logger';
import env from './env.json';

import Events from './telemetry/events';
import TelemetryClientContext from './telemetry/react_context';
import TelemetryClient from './telemetry/client';

import AppActions from './redux/actions/app';
import ConnectionActions from './redux/actions/connection';
import storeCreator from './redux/store';
import ConnectionStateView from './redux/views/connection_state';

export default class SkylinePlugin {
  constructor() {
    this._store = storeCreator();
    this._session = null;
    this._telemetryClient = TelemetryClient.from(env.uaId);

    this._getStartedClicked = this._getStartedClicked.bind(this);
    this._handleServerClosure = this._handleServerClosure.bind(this);
    this._handleInitializationTimeout = this._handleInitializationTimeout.bind(this);

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
          <PerfVis handleGetStartedClick={this._getStartedClicked} />
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
  }

  _disposeSession() {
    if (this._session == null) {
      return;
    }
    this._session.dispose();
    this._session = null;
    Logger.info('Disconnected from the server.');
  }

  _getStartedClicked({host, port}) {
    if (this._store.getState().appState !== AppState.OPENED) {
      return;
    }

    this._store.dispatch(ConnectionActions.connecting());
    this._session = new SkylineSession({
      store: this._store,
      telemetryClient: this._telemetryClient,
      handleServerClosure: this._handleServerClosure,
      handleInitializationTimeout: this._handleInitializationTimeout,
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
