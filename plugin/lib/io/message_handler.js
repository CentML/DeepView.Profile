'use babel';

import m from '../models_gen/messages_pb';
import PerfVisState from '../models/PerfVisState';
import INNPVStore from '../stores/innpv_store';
import OperationInfoStore from '../stores/operationinfo_store';
import BatchSizeStore from '../stores/batchsize_store';

export default class MessageHandler {
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
    }
  }
};
