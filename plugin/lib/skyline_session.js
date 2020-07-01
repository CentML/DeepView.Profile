'use babel';

import {CompositeDisposable} from 'atom';

import Connection from './io/connection';
import MessageHandler from './io/message_handler';
import MessageSender from './io/message_sender';
import AnalysisActions from './redux/actions/analysis';
import ConnectionActions from './redux/actions/connection';
import ConnectionStateView from './redux/views/connection_state';

export default class SkylineSession {
  constructor({store, telemetryClient, handleServerClosure, handleInitializationTimeout}) {
    this._handleMessage = this._handleMessage.bind(this);
    this._invokeTimeout = this._invokeTimeout.bind(this);
    this._handleInitializationTimeout = handleInitializationTimeout;

    this._disposed = false;
    this._disposables = new CompositeDisposable();

    this._store = store;
    const connectionStateView = new ConnectionStateView(this._store);
    this._telemetryClient = telemetryClient;

    this._connection = new Connection(this._handleMessage, handleServerClosure);
    this._messageSender = new MessageSender({
      connection: this._connection,
      connectionStateView,
      telemetryClient: this._telemetryClient,
    });
    this._messageHandler = new MessageHandler({
      messageSender: this._messageSender,
      connectionStateView,
      store: this._store,
      disposables: this._disposables,
      telemetryClient: this._telemetryClient,
    });
  }

  connect(host, port, timeoutMs = 5000) {
    return this._connection.connect(host, port)
      .then(() => {
        this._store.dispatch(ConnectionActions.initializing({
          onTimeout: this._handleInitializationTimeout,
        }));
        this._messageSender.sendInitializeRequest();
        setTimeout(this._invokeTimeout, timeoutMs);
      });
  }

  triggerProfiling() {
    if (this._disposed) {
      return;
    }
    this._store.dispatch(AnalysisActions.request());
    this._messageSender.sendAnalysisRequest();
  }

  dispose() {
    if (this._disposed) {
      return;
    }
    this._disposables.dispose();
    this._disposables = null;
    this._connection.close();
    this._connection = null;
    this._store = null;
    this._messageSender = null;
    this._messageHandler = null;
    this._disposed = true;
  }

  _handleMessage(message) {
    this._messageHandler.handleMessage(message);
  }

  _invokeTimeout() {
    if (this._store == null) {
      return;
    }

    const {onTimeout} = this._store.getState().connection;
    if (onTimeout == null) {
      return;
    }

    onTimeout();
  }
}
