'use babel';

export default class Throughput {
  constructor(samplesPerSecond, predictedMaxSamplesPerSecond) {
    this._samplesPerSecond = samplesPerSecond;
    this._predictedMaxSamplesPerSecond = predictedMaxSamplesPerSecond;
  }

  get hasMaxThroughputPrediction() {
    return !isNaN(this._predictedMaxSamplesPerSecond);
  }

  get samplesPerSecond() {
    return this._samplesPerSecond;
  }

  get predictedMaxSamplesPerSecond() {
    return this._predictedMaxSamplesPerSecond;
  }

  static fromThroughputResponse(throughputResponse) {
    return new Throughput(
      throughputResponse.getSamplesPerSecond(),
      throughputResponse.getPredictedMaxSamplesPerSecond(),
    );
  }
}
