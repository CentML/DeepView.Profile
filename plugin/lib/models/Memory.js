'use babel';

export default class Memory {
  constructor(usageMb, maxCapacityMb) {
    this._usageMb = usageMb;
    this._maxCapacityMb = maxCapacityMb;
  }

  get usageMb() {
    return this._usageMb;
  }

  get maxCapacityMb() {
    return this._maxCapacityMb;
  }

  get displayPct() {
    return this._usageMb / this._maxCapacityMb * 100;
  }

  static fromInfo(infoProtobuf) {
    return new Memory(
      infoProtobuf.getUsageMb(),
      infoProtobuf.getMaxCapacityMb(),
    );
  }

  static fromPrediction(infoProtobuf, batchSize) {
    const usageModel = infoProtobuf.getUsageModelMb();
    const predUsageMb = usageModel.getCoefficient() * batchSize + usageModel.getBias();
    return new Memory(
      predUsageMb,
      infoProtobuf.getMaxCapacityMb(),
    );
  }
}
