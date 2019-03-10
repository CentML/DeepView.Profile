'use babel';

export default class Memory {
  constructor(usage, maxCapacity) {
    this._usage = usage;
    this._maxCapacity = maxCapacity;
  }

  get usage() {
    return this._usage;
  }

  get maxCapacity() {
    return this._maxCapacity;
  }

  get displayPct() {
    return this._usage / this._maxCapacity * 100;
  }

  static fromInfo(infoProtobuf) {
    return new Memory(
      infoProtobuf.getUsage(),
      infoProtobuf.getMaxCapacity(),
    );
  }

  static fromPrediction(infoProtobuf, batchSize) {
    const capacityModel = infoProtobuf.getCapacityModel();
    const predUsage = capacityModel.getCoefficient() * batchSize + capacityModel.getBias();
    return new Memory(
      predUsage,
      infoProtobuf.getMaxCapacity(),
    );
  }
}
