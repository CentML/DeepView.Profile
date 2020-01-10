'use babel';

import pm from '../protocol_gen/innpv_pb';

export default class MessageSender {
  constructor(connection, connectionStateView) {
    this._connection = connection;
    this._connectionStateView = connectionStateView;
    this._initializeTimeout = null;
  }

  clearInitializeTimeout() {
    if (this._initializeTimeout == null) {
      return;
    }
    clearTimeout(this._initializeTimeout);
    this._initializeTimeout = null;
  }

  sendInitializeRequest(onTimeout, timeoutMs = 5000) {
    const message = new pm.InitializeRequest();
    // For now, we only have one version of the protocol
    message.setProtocolVersion(1);
    this._sendMessage(message, 'Initialize');
    this._initializeTimeout = setTimeout(onTimeout, timeoutMs);
  }

  sendAnalysisRequest() {
    const message = new pm.AnalysisRequest();
    this._sendMessage(message, 'Analysis');
  }

  _sendMessage(message, payloadName) {
    const enclosingMessage = new pm.FromClient();
    enclosingMessage['set' + payloadName](message);
    enclosingMessage.setSequenceNumber(
      this._connectionStateView.nextSequenceNumber(),
    );
    this._connection.sendBytes(enclosingMessage.serializeBinary());
  }
}
