'use babel';

import pm from '../protocol_gen/innpv_pb';

export default class MessageSender {
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
