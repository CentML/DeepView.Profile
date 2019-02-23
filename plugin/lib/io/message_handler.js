'use babel';

import m from '../models_gen/messages_pb';

export default class MessageHandler {
  constructor(messageSender) {
    this._messageSender = messageSender;
  }

  _handleAnalyzeResponse(message) {
    console.log('Received:', message.getResponse());
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
    }
  }
};
