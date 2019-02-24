'use babel';

import m from '../models_gen/messages_pb';

export default class MessageHandler {
  constructor(messageSender) {
    this._messageSender = messageSender;
  }

  _handleAnalyzeResponse(message) {
    const operationInfos = message.getResultsList();
    console.log('Received', operationInfos.length, 'messages.');
  }

  _handleAnalyzeError(message) {
    console.log('Received error message:', message.getErrorMessage());
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
