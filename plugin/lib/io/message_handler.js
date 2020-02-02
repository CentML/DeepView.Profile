'use babel';

import pm from '../protocol_gen/innpv_pb';

import INNPVFileTracker from '../editor/innpv_file_tracker';
import AnalysisActions from '../redux/actions/analysis';
import ConnectionActions from '../redux/actions/connection';
import Logger from '../logger';
import Events from '../telemetry/events';

export default class MessageHandler {
  constructor({messageSender, connectionStateView, store, disposables, telemetryClient}) {
    this._messageSender = messageSender;
    this._connectionStateView = connectionStateView;
    this._store = store;
    this._disposables = disposables;
    this._telemetryClient = telemetryClient;

    this._handleInitializeResponse = this._handleInitializeResponse.bind(this);
    this._handleProtocolError = this._handleProtocolError.bind(this);
    this._handleMemoryUsageResponse = this._handleMemoryUsageResponse.bind(this);
    this._handleAnalysisError = this._handleAnalysisError.bind(this);
    this._handleThroughputResponse = this._handleThroughputResponse.bind(this);
    this._handleRunTimeResponse = this._handleRunTimeResponse.bind(this);
    this._handleBreakdownResponse = this._handleBreakdownResponse.bind(this);
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
    this._telemetryClient.record(Events.Skyline.CONNECTED);
    Logger.info('Connected!');

    Logger.info('Sending analysis request...');
    this._store.dispatch(AnalysisActions.request());
    this._messageSender.sendAnalysisRequest();
  }

  _handleProtocolError(message) {
    const errorCode = message.getErrorCode();
    if (errorCode === pm.ProtocolError.ErrorCode.UNSUPPORTED_PROTOCOL_VERSION) {
      Logger.error(
        'The plugin and/or server that you are using are out of date. ' +
        'Please update them before retrying.',
      );
      return;
    }
    Logger.error(
      `Received a protocol error with code: ${errorCode}. Please file a bug report.`);
    this._telemetryClient.record(Events.Error.PROTOCOL_ERROR);
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
    this._telemetryClient.record(Events.Error.ANALYSIS_ERROR);
  }

  _handleThroughputResponse(message) {
    Logger.info(`Received throughput message: ${message.getSamplesPerSecond()} samples/s.`);
    this._store.dispatch(AnalysisActions.receivedThroughputAnalysis({
      throughputResponse: message,
    }));
    this._telemetryClient.record(Events.Skyline.RECEIVED_ANALYSIS);
  }

  _handleRunTimeResponse(message) {
    Logger.info('Received run time message.');
    this._store.dispatch(AnalysisActions.receivedRunTimeAnalysis({
      runTimeResponse: message,
    }));
  }

  _handleBreakdownResponse(message) {
    Logger.info('Received breakdown message.');
    this._store.dispatch(AnalysisActions.receivedBreakdown({
      breakdownResponse: message,
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

      case pm.FromServer.PayloadCase.BREAKDOWN:
        this._handleAfterInitializationMessage(
          this._handleBreakdownResponse,
          enclosingMessage.getBreakdown(),
        );
        break;
    }
  }
}
