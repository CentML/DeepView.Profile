'use babel';

import pm from '../protocol_gen/innpv_pb';
import AppState from '../models/AppState';
import PerfVisState from '../models/PerfVisState';
import INNPVStore from '../stores/innpv_store';
import OperationInfoStore from '../stores/operationinfo_store';
import BatchSizeStore from '../stores/batchsize_store';

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
