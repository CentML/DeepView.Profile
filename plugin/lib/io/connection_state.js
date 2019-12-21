'use babel';

export default class ConnectionState {
  constructor() {
    this._analysisSequenceNumber = 0;
    // The connection only has two states: uninitialized and "ready" (initialized)
    // Therefore for simplicity, we use a boolean to represent these states
    this._initialized = false;
    this._fileTracker = null;
  }

  dispose() {
    if (this._fileTracker != null) {
      this._fileTracker.dispose();
      this._fileTracker = null;
    }
  }

  get initialized() {
    return this._initialized;
  }

  markInitialized(fileTracker) {
    this._initialized = true;
    this._fileTracker = fileTracker;
  }

  nextAnalysisSequenceNumber() {
    const number = this._analysisSequenceNumber;
    this._analysisSequenceNumber += 1;
    return number;
  }

  isResponseCurrent(responseSequenceNumber) {
    // Since we always increase the sequence number by one, a "current"
    // response is one with a sequence number exactly one less than the next
    // sequence number to be assigned (this._analysisSequenceNumber).
    return responseSequenceNumber === this._analysisSequenceNumber - 1;
  }
}
