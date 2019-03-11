'use babel';

import BaseStore from './base_store';
import Throughput from '../models/Throughput';
import Memory from '../models/Memory';
import {
  evaluateLinearModel,
  getBatchSizeFromUsage,
  getBatchSizeFromThroughput,
} from '../utils';

class BatchSizeStore extends BaseStore {
  constructor() {
    super();
    this._throughputInfo = null;
    this._memoryInfo = null;
    this._batchSize = null;
    this._predictedBatchSize = null;
    this._maxBatchSize = null;
  }

  setInfos(throughputInfo, memoryInfo, batchSize) {
    this._throughputInfo = throughputInfo;
    this._memoryInfo = memoryInfo;
    this._batchSize = batchSize;
    this._predictedBatchSize = null;
    this._maxBatchSize = getBatchSizeFromUsage(
      this._memoryInfo.getUsageModelMb(),
      this._memoryInfo.getMaxCapacityMb(),
    );
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
    this._predictedBatchSize = Math.max(
      getBatchSizeFromUsage(this._memoryInfo.getUsageModelMb(), updatedUsage),
      1,
    );
    this.notifyChanged();
    return this._predictedBatchSize;
  }

  updateThroughput(deltaPct, basePct) {
    // Map the delta to a throughput value
    // NOTE: We clamp the values (upper bound for throughput, lower bound for batch size)
    const updatedPct = basePct + deltaPct;
    const updatedThroughput = Math.max(Math.min(
      updatedPct / 100 * this._throughputInfo.getMaxThroughput(),
      this._throughputInfo.getThroughputLimit(),
    ), 0);
    const throughputBatchSize = getBatchSizeFromThroughput(
      this._throughputInfo.getRuntimeModelMs(),
      updatedThroughput,
    );

    if (throughputBatchSize < 0) {
      // NOTE: The throughput batch size may be so large that it overflows
      this._predictedBatchSize = this._maxBatchSize;
    } else {
      this._predictedBatchSize = Math.max(Math.min(throughputBatchSize, this._maxBatchSize), 1);
    }

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
