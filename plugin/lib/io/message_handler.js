'use babel';

import m from '../models_gen/messages_pb';
import pm from '../protocol_gen/innpv_pb';
import AppState from '../models/AppState';
import PerfVisState from '../models/PerfVisState';
import INNPVStore from '../stores/innpv_store';
import OperationInfoStore from '../stores/operationinfo_store';
import BatchSizeStore from '../stores/batchsize_store';

class MessageHandler {
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
    INNPVStore.setPaths(message.getServerProjectRoot(), message.getEntryPoint());
    INNPVStore.clearErrorMessage();
    INNPVStore.setAppState(AppState.CONNECTED);
    this._connectionState.markInitialized();
    console.log('Connected!');
  }

  _handleProtocolError(message) {
    console.error(`Received a protocol error with code: ${message.getErrorCode()}`);
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
    }
  }
}

class LegacyMessageHandler {
  constructor(messageSender, protocol) {
    this._messageSender = messageSender;
    this._protocol = protocol;
  }

  _handleAnalyzeResponse(message) {
    const sequenceNumber = message.getSequenceNumber();
    if (!this._protocol.isResponseCurrent(sequenceNumber)) {
      console.log('Ignoring stale analyze response message with sequence number:', sequenceNumber);
      return;
    }

    console.log('Received response with sequence number:', sequenceNumber);
    const operationInfos = message.getResultsList();
    OperationInfoStore.setOperationInfos(operationInfos);
    BatchSizeStore.receivedAnalysis(
      message.getThroughput(),
      message.getMemory(),
      message.getInput(),
      message.getLimits(),
    );
    INNPVStore.setPerfVisState(PerfVisState.READY);
    INNPVStore.clearErrorMessage();
  }

  _handleAnalyzeError(message) {
    const sequenceNumber = message.getSequenceNumber();
    if (!this._protocol.isResponseCurrent(sequenceNumber)) {
      console.log('Ignoring stale analyze error message with sequence number:', sequenceNumber);
      return;
    }

    console.log('Received error message with sequence number:', sequenceNumber);
    INNPVStore.setErrorMessage(message.getErrorMessage());
    INNPVStore.setPerfVisState(PerfVisState.ERROR);
  }

  _handleProfiledLayersResponse(message) {
    // Protocol dictates that this message is received 1st
    const sequenceNumber = message.getSequenceNumber();
    if (!this._protocol.isResponseCurrent(sequenceNumber)) {
      console.log('Ignoring stale profiled layers response with sequence number:', sequenceNumber);
      return;
    }

    console.log('Received profiled layers response message with sequence number:', sequenceNumber);
    OperationInfoStore.setOperationInfos(message.getResultsList());
    INNPVStore.clearErrorMessage();
  }

  _handleMemoryInfoResponse(message) {
    // Protocol dictates that this message is received 2nd
    const sequenceNumber = message.getSequenceNumber();
    if (!this._protocol.isResponseCurrent(sequenceNumber)) {
      console.log('Ignoring stale memory info response with sequence number:', sequenceNumber);
      return;
    }

    console.log('Received memory info response message with sequence number:', sequenceNumber);
    BatchSizeStore.receivedMemoryResponse(message.getMemory(), message.getInput());
  }

  _handleThroughputInfoResponse(message) {
    // Protocol dictates that this message is received 3rd
    const sequenceNumber = message.getSequenceNumber();
    if (!this._protocol.isResponseCurrent(sequenceNumber)) {
      console.log('Ignoring stale throughput info response with sequence number:', sequenceNumber);
      return;
    }

    console.log('Received throughput info response message with sequence number:', sequenceNumber);
    BatchSizeStore.receivedThroughputResponse(message.getThroughput(), message.getLimits());
    INNPVStore.setPerfVisState(PerfVisState.READY);
  }

  handleMessage(byteArray) {
    const enclosingMessage = m.ServerMessage.deserializeBinary(byteArray);
    const payloadCase = enclosingMessage.getPayloadCase();

    switch (payloadCase) {
      case m.ServerMessage.PayloadCase.PAYLOAD_NOT_SET:
        console.warn('Received an empty message from the server.');
        break;

      case m.ServerMessage.PayloadCase.ANALYZE_RESPONSE:
        this._handleAnalyzeResponse(enclosingMessage.getAnalyzeResponse());
        break;

      case m.ServerMessage.PayloadCase.ANALYZE_ERROR:
        this._handleAnalyzeError(enclosingMessage.getAnalyzeError());
        break;

      case m.ServerMessage.PayloadCase.PROFILED_LAYERS_RESPONSE:
        this._handleProfiledLayersResponse(enclosingMessage.getProfiledLayersResponse());
        break;

      case m.ServerMessage.PayloadCase.MEMORY_INFO_RESPONSE:
        this._handleMemoryInfoResponse(enclosingMessage.getMemoryInfoResponse());
        break;

      case m.ServerMessage.PayloadCase.THROUGHPUT_INFO_RESPONSE:
        this._handleThroughputInfoResponse(enclosingMessage.getThroughputInfoResponse());
        break;
    }
  }
};

export default MessageHandler;
