'use babel';

import m from '../models_gen/messages_pb';
import PerfVisState from '../models/PerfVisState';
import INNPVStore from '../stores/innpv_store';
import OperationInfoStore from '../stores/operationinfo_store';
import BatchSizeStore from '../stores/batchsize_store';

export default class MessageHandler {
  constructor(messageSender) {
    this._messageSender = messageSender;
  }

  _handleAnalyzeResponse(message) {
    const operationInfos = message.getResultsList();
    console.log('Received', operationInfos.length, 'messages.');
    const artificalDelay = () => {
      OperationInfoStore.setOperationInfos(operationInfos);
      BatchSizeStore.setInfos(message.getThroughput(), null, message.getBatchSize());
      INNPVStore.setPerfVisState(PerfVisState.READY);
      INNPVStore.clearErrorMessage();
    };
    setTimeout(artificalDelay, 1500);
  }

  _handleAnalyzeError(message) {
    console.log('Received error message:', message.getErrorMessage());
    const artificalDelay = () => {
      INNPVStore.setErrorMessage(message.getErrorMessage());
      INNPVStore.setPerfVisState(PerfVisState.ERROR);
    };
    setTimeout(artificalDelay, 1500);
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
