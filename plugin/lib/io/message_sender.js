'use babel';

import m from '../models_gen/messages_pb';

export default class MessageSender {
  constructor(connection, protocol) {
    this._connection = connection;
    this._protocol = protocol;
  }

  sendAnalyzeRequest(sourceCode) {
    const message = new m.AnalyzeRequest();
    message.setSourceCode(sourceCode);
    // message.setMockResponse(true);
    message.setSequenceNumber(this._protocol.nextAnalysisSequenceNumber());
    this._sendMessage(message, 'AnalyzeRequest');
  }

  _sendMessage(message, payload_name) {
    const enclosingMessage = new m.PluginMessage();
    enclosingMessage['set' + payload_name](message);
    this._connection.sendBytes(enclosingMessage.serializeBinary());
  }
}
