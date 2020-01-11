'use babel';

import pm from '../protocol_gen/innpv_pb';

import INNPVFileTracker from '../editor/innpv_file_tracker';
import AnalysisActions from '../redux/actions/analysis';
import ConnectionActions from '../redux/actions/connection';
import Logger from '../logger';

export default class MessageHandler {
  constructor(messageSender, connectionStateView, store, disposables) {
    this._messageSender = messageSender;
    this._connectionStateView = connectionStateView;
    this._store = store;
    this._disposables = disposables;

    this._handleInitializeResponse = this._handleInitializeResponse.bind(this);
    this._handleProtocolError = this._handleProtocolError.bind(this);
    this._handleMemoryUsageResponse = this._handleMemoryUsageResponse.bind(this);
    this._handleAnalysisError = this._handleAnalysisError.bind(this);
    this._handleThroughputResponse = this._handleThroughputResponse.bind(this);
  }

  _handleInitializeResponse(message) {
    if (this._connectionStateView.isInitialized()) {
      Logger.warn('Connection already initialized, but received an initialize response.');
      return;
    }

    // TODO: Validate the project root and entry point paths.
    //       We don't (yet) support remote work, so "trusting" the server is fine
    //       for now because we validate the paths on the server.
    const projectRoot = message.getServerProjectRoot();

    this._store.dispatch(ConnectionActions.initialized({projectRoot}));
    this._disposables.add(new INNPVFileTracker(projectRoot, this._messageSender, this._store));
    Logger.info('Connected!');

    Logger.info('Sending analysis request...');
    this._store.dispatch(AnalysisActions.request());
    this._messageSender.sendAnalysisRequest();
  }

  _handleProtocolError(message) {
    const errorCode = message.getErrorCode();
    if (errorCode === pm.ProtocolError.ErrorCode.UNSUPPORTED_PROTOCOL_VERSION) {
      Logger.error('The plugin that you are using is out of date. Please update it before retrying.');
      return;
    }
    Logger.error(
      `Received a protocol error with code: ${errorCode}. Please file a bug report.`);
  }

  _handleMemoryUsageResponse(message) {
    Logger.info('Received memory usage message.');
    Logger.info(`Peak usage: ${message.getPeakUsageBytes()} bytes.`);
    this._store.dispatch(AnalysisActions.receivedMemoryAnalysis({
      memoryUsageResponse: message,
    }));
  }

  _handleAnalysisError(message) {
    this._store.dispatch(AnalysisActions.error({
      errorMessage: message.getErrorMessage(),
    }));
  }

  _handleThroughputResponse(message) {
    Logger.info(`Received throughput message: ${message.getSamplesPerSecond()} samples/s.`);
    this._store.dispatch(AnalysisActions.receivedThroughputAnalysis({
      throughputResponse: message,
    }));
  }

  _handleAfterInitializationMessage(handler, message) {
    if (!this._connectionStateView.isInitialized()) {
      Logger.warn('Connection not initialized, but received a regular protocol message.');
      return;
    }
    handler(message);
  }

  handleMessage(byteArray) {
    const enclosingMessage = pm.FromServer.deserializeBinary(byteArray);
    const payloadCase = enclosingMessage.getPayloadCase();

    if (!this._connectionStateView.isResponseCurrent(enclosingMessage.getSequenceNumber())) {
      // Ignore old responses (e.g., if we make a new analysis request before
      // the previous one completes).
      Logger.info('Ignoring stale response with sequence number:', enclosingMessage.getSequenceNumber());
      return;
    }

    switch (payloadCase) {
      case pm.FromServer.PayloadCase.PAYLOAD_NOT_SET:
        Logger.warn('Received an empty message from the server.');
        break;

      case pm.FromServer.PayloadCase.INITIALIZE:
        this._handleInitializeResponse(enclosingMessage.getInitialize());
        break;

      case pm.FromServer.PayloadCase.ERROR:
        this._handleProtocolError(enclosingMessage.getError());
        break;

      case pm.FromServer.PayloadCase.MEMORY_USAGE:
        this._handleAfterInitializationMessage(
          this._handleMemoryUsageResponse,
          enclosingMessage.getMemoryUsage(),
        );
        break;

      case pm.FromServer.PayloadCase.ANALYSIS_ERROR:
        this._handleAfterInitializationMessage(
          this._handleAnalysisError,
          enclosingMessage.getAnalysisError(),
        );
        break;

      case pm.FromServer.PayloadCase.THROUGHPUT:
        this._handleAfterInitializationMessage(
          this._handleThroughputResponse,
          enclosingMessage.getThroughput(),
        );
        break;
    }
  }
}
