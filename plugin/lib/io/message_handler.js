'use babel';

import pm from '../protocol_gen/innpv_pb';
import AppState from '../models/AppState';
import PerfVisState from '../models/PerfVisState';
import INNPVStore from '../stores/innpv_store';
import AnalysisStore from '../stores/analysis_store';
import INNPVFileTracker from '../editor/innpv_file_tracker';
import Logger from '../logger';

export default class MessageHandler {
  constructor(messageSender, connectionState) {
    this._messageSender = messageSender;
    this._connectionState = connectionState;
  }

  _handleInitializeResponse(message) {
    if (this._connectionState.initialized) {
      Logger.warn('Connection already initialized, but received an initialize response.');
      return;
    }

    // TODO: Validate the project root and entry point paths.
    //       We don't (yet) support remote work, so "trusting" the server is fine
    //       for now because we validate the paths on the server.
    INNPVStore.clearErrorMessage();
    INNPVStore.setAppState(AppState.CONNECTED);
    this._connectionState.markInitialized(
      new INNPVFileTracker(message.getServerProjectRoot(), this._messageSender),
    );
    Logger.info('Connected!');

    Logger.info('Sending analysis request...');
    INNPVStore.setPerfVisState(PerfVisState.ANALYZING);
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
    AnalysisStore.receivedMemoryUsage(message);
    INNPVStore.clearErrorMessage();
  }

  _handleAnalysisError(message) {
    INNPVStore.setErrorMessage(message.getErrorMessage());
    INNPVStore.setPerfVisState(PerfVisState.ERROR);
  }

  _handleThroughputResponse(message) {
    Logger.info(`Received throughput message: ${message.getSamplesPerSecond()} samples/s.`);
    AnalysisStore.receivedThroughput(message);
    INNPVStore.clearErrorMessage();
    if (INNPVStore.getPerfVisState() !== PerfVisState.MODIFIED) {
      INNPVStore.setPerfVisState(PerfVisState.READY);
    }
  }

  _handleAfterInitializationMessage(handler, message) {
    if (!this._connectionState.initialized) {
      Logger.warn('Connection not initialized, but received a regular protocol message.');
      return;
    }
    handler(message);
  }

  handleMessage(byteArray) {
    const enclosingMessage = pm.FromServer.deserializeBinary(byteArray);
    const payloadCase = enclosingMessage.getPayloadCase();

    if (!this._connectionState.isResponseCurrent(enclosingMessage.getSequenceNumber())) {
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
