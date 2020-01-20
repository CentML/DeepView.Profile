'use babel';

import pm from '../protocol_gen/innpv_pb';
import Events from '../telemetry/events';

export default class MessageSender {
  constructor({connection, connectionStateView, telemetryClient}) {
    this._connection = connection;
    this._connectionStateView = connectionStateView;
    this._telemetryClient = telemetryClient;
  }

  sendInitializeRequest() {
    const message = new pm.InitializeRequest();
    // For now, we only have one version of the protocol
    message.setProtocolVersion(1);
    this._sendMessage(message, 'Initialize');
  }

  sendAnalysisRequest() {
    const message = new pm.AnalysisRequest();
    this._sendMessage(message, 'Analysis');
    this._telemetryClient.record(Events.Skyline.REQUESTED_ANALYSIS);
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
