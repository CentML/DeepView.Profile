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

  static fromInfo(infoProtobuf) {
    return new Throughput(
      infoProtobuf.getThroughput(),
      infoProtobuf.getMaxThroughput(),
      infoProtobuf.getThroughputLimit(),
    );
  }

  static fromPrediction(infoProtobuf, batchSize) {
    const runtimeModel = infoProtobuf.getRuntimeModelMs();
    const predRuntime = runtimeModel.getCoefficient() * batchSize + runtimeModel.getBias();
    const predThroughput = batchSize / predRuntime;
    return new Throughput(
      predThroughput,
      infoProtobuf.getMaxThroughput(),
      infoProtobuf.getThroughputLimit(),
    );
  }
}
