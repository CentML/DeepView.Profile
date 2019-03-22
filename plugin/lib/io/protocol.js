'use babel';

export default class Protocol {
  constructor() {
    this._analysisSequenceNumber = 0;
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
