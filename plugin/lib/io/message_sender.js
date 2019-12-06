'use babel';

import m from '../models_gen/messages_pb';
import pm from '../protocol_gen/innpv_pb';

class MessageSender {
  constructor(connection, connectionState) {
    this._connection = connection;
    this._connectionState = connectionState;
  }

  sendInitializeRequest() {
    const message = new pm.InitializeRequest();
    // For now, we only have one version of the protocol
    message.setProtocolVersion(1);
    this._sendMessage(message, 'Initialize');
  }

  _sendMessage(message, payloadName) {
    const enclosingMessage = new pm.FromClient();
    enclosingMessage['set' + payloadName](message);
    this._connection.sendBytes(enclosingMessage.serializeBinary());
  }
}

class LegacyMessageSender {
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

export default MessageSender;
