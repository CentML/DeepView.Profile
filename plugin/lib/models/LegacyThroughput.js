'use babel';

export default class Throughput {
  constructor(throughput, maxThroughput, throughputLimit) {
    this._maxThroughput = maxThroughput;
    this._throughput = throughput;
    this._throughputLimit = throughputLimit;
  }

  get throughput() {
    return this._throughput;
  }

  get maxThroughput() {
    return this._maxThroughput;
  }

  get displayPct() {
    return this._throughput / this._maxThroughput * 100;
  }

  get limitPct() {
    return this._throughputLimit / this._maxThroughput * 100;
  }

  static fromInfo(infoProtobuf, limitsProtobuf) {
    return new Throughput(
      infoProtobuf.getThroughput(),
      infoProtobuf.getMaxThroughput(),
      limitsProtobuf.getThroughputLimit(),
    );
  }

  static fromPrediction(infoProtobuf, limitsProtobuf, batchSize) {
    const runtimeModel = infoProtobuf.getRuntimeModelMs();
    const predRuntime = runtimeModel.getCoefficient() * batchSize + runtimeModel.getBias();
    // Runtime is in milliseconds, so multiply throughput by 1000 to get units in seconds
    const predThroughput = batchSize / predRuntime * 1000;
    return new Throughput(
      predThroughput,
      infoProtobuf.getMaxThroughput(),
      limitsProtobuf.getThroughputLimit(),
    );
  }
}
