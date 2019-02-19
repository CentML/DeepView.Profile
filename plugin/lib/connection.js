'use babel';

import net from 'net';
import { CompositeDisposable } from 'atom';

export default class Connection {
  constructor() {
    this._socket = net.createConnection(6060, 'localhost', () => {
      console.log('Connected!');
      this._socket.write('Hello world!\n');
    });
    this._socket.setEncoding('utf8');
    this._handleData = this._handleData.bind(this);

    this._socket.on('data', this._handleData);
    this._socket.once('end', () => {
      console.log('Connection closed by the server!');
      this.close();
    });
  }

  close() {
    if (this._socket == null) {
      return;
    }
    this._socket.removeListener('data', this._handleData);
    this._socket.end();
    this._socket = null;
    console.log('Connection closed.');
  }

  _handleData(data) {
    console.log('Received message:', data);
  }
}
