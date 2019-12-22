'use babel';

import pm from '../protocol_gen/innpv_pb';
import AppState from '../models/AppState';
import PerfVisState from '../models/PerfVisState';
import INNPVStore from '../stores/innpv_store';
import AnalysisStore from '../stores/analysis_store';
import INNPVFileTracker from '../editor/innpv_file_tracker';

export default class MessageHandler {
  constructor(messageSender, connectionState) {
    this._messageSender = messageSender;
    this._connectionState = connectionState;
  }

  _handleInitializeResponse(message) {
    if (this._connectionState.initialized) {
      console.warn('Connection already initialized, but received an initialize response.');
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
    console.log('Connected!');

    console.log('Sending analysis request...');
    INNPVStore.setPerfVisState(PerfVisState.ANALYZING);
    this._messageSender.sendAnalysisRequest();
  }

  _handleProtocolError(message) {
    console.error(`Received a protocol error with code: ${message.getErrorCode()}`);
  }

  _handleMemoryUsageResponse(message) {
    console.log('Received memory usage message.');
    console.log(`Peak usage: ${message.getPeakUsageBytes()} bytes.`);
    AnalysisStore.receivedMemoryUsage(message);
    INNPVStore.clearErrorMessage();
  }

  _handleAnalysisError(message) {
    INNPVStore.setErrorMessage(message.getErrorMessage());
    INNPVStore.setPerfVisState(PerfVisState.ERROR);
  }

  _handleThroughputResponse(message) {
    console.log(`Received throughput message: ${message.getSamplesPerSecond()} samples/s.`);
    INNPVStore.clearErrorMessage();
    if (INNPVStore.getPerfVisState() !== PerfVisState.MODIFIED) {
      INNPVStore.setPerfVisState(PerfVisState.READY);
    }
  }

  _handleAfterInitializationMessage(handler, message) {
    if (!this._connectionState.initialized) {
      console.warn('Connection not initialized, but received a regular protocol message.');
    }
    if (!this._connectionState.isResponseCurrent(message.getSequenceNumber())) {
      // Ignore old responses (e.g., if we make a new analysis request before
      // the previous one completes).
      return;
    }
    handler(message);
  }

  handleMessage(byteArray) {
    const enclosingMessage = pm.FromServer.deserializeBinary(byteArray);
    const payloadCase = enclosingMessage.getPayloadCase();

    switch (payloadCase) {
      case pm.FromServer.PayloadCase.PAYLOAD_NOT_SET:
        console.warn('Received an empty message from the server.');
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
