'use babel';

import BaseStore from './base_store';
import Throughput from '../models/Throughput';
import Memory from '../models/Memory';

class BatchSizeStore extends BaseStore {
  constructor() {
    super();
    this._throughputInfo = null;
    this._memoryInfo = null;
    this._batchSize = null;
    this._predictedBatchSize = null;
  }

  setInfos(throughputInfo, memoryInfo, batchSize) {
    this._throughputInfo = throughputInfo;
    this._memoryInfo = memoryInfo;
    this._batchSize = batchSize;
    this._predictedBatchSize = null;
    this.notifyChanged();
  }

  updateMemoryUsage(deltaPct, basePct) {
    // Map the delta to a usage value
    // NOTE: We clamp the values (upper bound for usage, lower bound for batch size)
    const updatedPct = basePct + deltaPct;
    const updatedUsage = Math.min(
      updatedPct / 100 * this._memoryInfo.getMaxCapacityMb(),
      this._memoryInfo.getMaxCapacityMb(),
    );
    const model = this._memoryInfo.getUsageModelMb();
    this._predictedBatchSize = Math.max((updatedUsage - model.getBias()) / model.getCoefficient(), 1);
    this.notifyChanged();
    return this._predictedBatchSize;
  }

  getThroughputModel() {
    if (this._throughputInfo == null) {
      return null;
    }

    if (this._predictedBatchSize == null) {
      return Throughput.fromInfo(this._throughputInfo);
    } else {
      return Throughput.fromPrediction(this._throughputInfo, this._predictedBatchSize);
    }
  }

  getMemoryModel() {
    if (this._memoryInfo == null) {
      return null;
    }

    if (this._predictedBatchSize == null) {
      return Memory.fromInfo(this._memoryInfo);
    } else {
      return Memory.fromPrediction(this._memoryInfo, this._predictedBatchSize);
    }
  }
}

const storeInstance = new BatchSizeStore();

export default storeInstance;
