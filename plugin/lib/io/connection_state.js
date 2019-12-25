'use babel';

export default class ConnectionState {
  constructor() {
    this._analysisSequenceNumber = 0;
    // The connection only has two states: uninitialized and "ready" (initialized)
    // Therefore for simplicity, we use a boolean to represent these states
    this._initialized = false;
    this._fileTracker = null;
    this._initializationTimeout = null;
  }

  dispose() {
    if (this._fileTracker != null) {
      this._fileTracker.dispose();
      this._fileTracker = null;
    }
    if (this._initializationTimeout != null) {
      clearTimeout(this._initializationTimeout);
      this._initializationTimeout = null;
    }
  }

  get initialized() {
    return this._initialized;
  }

  setInitializationTimeout(fn, timeoutMs = 5000) {
    this._initializationTimeout = setTimeout(fn, timeoutMs);
  }

  markInitialized(fileTracker) {
    this._initialized = true;
    this._fileTracker = fileTracker;

    if (this._initializationTimeout != null) {
      clearTimeout(this._initializationTimeout);
      this._initializationTimeout = null;
    }
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
