'use babel';

import net from 'net';
import { CompositeDisposable } from 'atom';
import Logger from '../logger';

export default class Connection {
  constructor(dataHandler, onServerClosure) {
    this._connected = false;
    this._socket = new net.Socket();
    this._dataHandler = dataHandler;
    this._onServerClosure = onServerClosure;

    // Used for the incoming message
    this._incomingBuffers = [];
    this._nextMessageLength = -1;

    this._handleData = this._handleData.bind(this);
    this._handleServerClosure = this._handleServerClosure.bind(this);
  }

  connect(host, port) {
    if (this._connected) {
      return Promise.resolve();
    }
    if (this._connectingPromise != null) {
      return this._connectingPromise;
    }

    this._connectingPromise = new Promise((resolve, reject) => {
      this._socket.once('error', (error) => {
        this._connectingPromise = null;
        reject(error);
      });

      this._socket.connect({host, port}, () => {
        this._socket.removeListener('error', reject);

        this._socket.on('data', this._handleData);
        this._socket.once('end', this._handleServerClosure);

        this._connectingPromise = null;
        this._connected = true;
        resolve();
      });
    });

    return this._connectingPromise;
  }

  close() {
    if (!this._connected) {
      return;
    }
    this._socket.removeAllListeners();
    this._socket.destroy();
    this._socket = null;
    this._connected = false;
  }

  sendBytes(byteArray) {
    // 32-bit unsigned integer for the message length
    // NOTE: Use big endian to respect network byte order
    const lengthBuffer = Buffer.alloc(4);
    lengthBuffer.writeUInt32BE(byteArray.length, 0);
    this._socket.write(lengthBuffer);
    this._socket.write(Buffer.from(byteArray));
  }

  _handleServerClosure() {
    // Called when the server initiates the connection closure
    Logger.info('Connection closure initiated by the server.');
    if (this._onServerClosure != null) {
      this._onServerClosure();
    }
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

    this._nextMessageLength = -1;

    this._dataHandler(messageResult);
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
    Logger.error('Reached invalid spot in Connection._readIntoBuffer()');
    return null;
  }
}
