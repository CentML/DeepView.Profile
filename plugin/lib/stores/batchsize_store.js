'use babel';

import BaseStore from './base_store';
import Throughput from '../models/Throughput';

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
    this.notifyChanged();
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
}

const storeInstance = new BatchSizeStore();

export default storeInstance;
