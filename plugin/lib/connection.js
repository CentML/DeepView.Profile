'use babel';

import net from 'net';
import { CompositeDisposable } from 'atom';
import m from './models_gen/messages_pb';

export default class Connection {
  constructor() {
    this._socket = net.createConnection(6060, 'localhost', () => {
      console.log('Connected!');
      const msg = new m.AnalyzeRequest();
      msg.setSourceCode('Hello world!');
      this._sendMessage(msg);
    });

    // Used for the incoming message
    this._incomingBuffers = [];
    this._nextMessageLength = -1;

    // Socket event bindings
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

  _sendMessage(message) {
    const messageBuffer = message.serializeBinary();
    // 32-bit unsigned integer for the message length
    // NOTE: Use big endian to respect network byte order
    const lengthBuffer = Buffer.alloc(4);
    lengthBuffer.writeUInt32BE(messageBuffer.length, 0);
    this._socket.write(lengthBuffer);
    this._socket.write(Buffer.from(messageBuffer));
  }

  _handleData(chunk) {
    this._incomingBuffers.push(chunk);

    if (this._nextMessageLength <= 0) {
      // We have not received a complete message length yet
      const lengthResult = this._readIntoBuffer(4);
      if (lengthResult === null) {
        return;
      }
      this._nextMessageLength = lengthResult.readUInt32BE();
    }

    const messageResult = this._readIntoBuffer(this._nextMessageLength);
    if (messageResult === null) {
      return;
    }

    const message = m.AnalyzeResponse.deserializeBinary(messageResult);
    this._nextMessageLength = -1;

    this._handleMessage(message);
  }

  _readIntoBuffer(size) {
    const totalBytes = this._incomingBuffers.reduce((accum, buf) => accum + buf.length, 0);
    if (totalBytes < size) {
      return null;
    }

    const numBuffers = this._incomingBuffers.length;
    const buffer = Buffer.alloc(size);
    let offset = 0;
    let bytesLeft = size;

    // NOTE: Loop at most numBuffers times to prevent an infinite loop
    //       if there is a bug. However if implemented correctly this
    //       loop should terminate even if it was a while (true) {}.
    for (let i = 0; i < numBuffers; i++) {
      const inputBuffer = this._incomingBuffers[0];
      inputBuffer.copy(buffer, offset, 0, bytesLeft);

      if (inputBuffer.length == bytesLeft) {
        this._incomingBuffers.shift()
        return buffer;

      } else if (inputBuffer.length > bytesLeft) {
        const leftoverBytes = inputBuffer.length - bytesLeft;
        const replacementBuffer = Buffer.alloc(leftoverBytes);
        inputBuffer.copy(replacementBuffer, 0, bytesLeft);
        this._incomingBuffers[0] = replacementBuffer;
        return buffer;

      } else {
        offset += inputBuffer.length;
        bytesLeft -= inputBuffer.length;
        this._incomingBuffers.shift();
      }
    }

    // NOTE: We should never reach here!
    console.error('Reached invalid spot in Connection._readIntoBuffer()');
    return null;
  }

  _handleMessage(message) {
    console.log('Received response:', message.getResponse());
  }
}
