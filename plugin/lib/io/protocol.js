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
}
