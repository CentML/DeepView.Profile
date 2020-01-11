'use babel';

import BaseStore from './base_store';
import Throughput from '../models/LegacyThroughput';
import Memory from '../models/Memory';
import {
  evaluateLinearModel,
  getBatchSizeFromUsage,
  getBatchSizeFromThroughput,
} from '../utils';
import INNPVStore from './innpv_store';
import {Range} from 'atom';

class BatchSizeStore extends BaseStore {
  constructor() {
    super();
  }

  reset() {
    this._throughputInfo = null;
    this._memoryInfo = null;
    this._inputInfo = null;
    this._perfLimits = null;

    this._predictedBatchSize = null;
    this._maxBatchSize = null;

    this._currentAnnotationRange = null;

    this._clearViewDebounce = null;
  }

  receivedAnalysis(throughputInfo, memoryInfo, inputInfo, perfLimits) {
    this._cancelClearView();
    this._throughputInfo = throughputInfo;
    this._memoryInfo = memoryInfo;
    this._inputInfo = inputInfo;
    this._perfLimits = perfLimits;

    this._updateComputedState();
    this.notifyChanged();
  }

  receivedMemoryResponse(memoryInfo, inputInfo) {
    this._cancelClearView();
    this._memoryInfo = memoryInfo;
    this._inputInfo = inputInfo;
    this.notifyChanged();
  }

  receivedThroughputResponse(throughputInfo, perfLimits) {
    this._cancelClearView();
    this._throughputInfo = throughputInfo;
    this._perfLimits = perfLimits;
    this._updateComputedState();
    this.notifyChanged();
  }

  _updateComputedState() {
    this._predictedBatchSize = null;
    this._maxBatchSize = this._perfLimits.getMaxBatchSize();

    const startPoint = this._inputInfo.getAnnotationStart();
    const endPoint = this._inputInfo.getAnnotationEnd();
    this._currentAnnotationRange = new Range(
      [startPoint.getLine(), startPoint.getColumn()],
      [endPoint.getLine(), endPoint.getColumn()],
    );
  }

  updateMemoryUsage(deltaPct, basePct) {
    // Map the delta to a usage value
    // NOTE: We clamp the values (upper bound for usage, lower bound for batch size)
    const updatedPct = basePct + deltaPct;
    const updatedUsage = Math.min(
      updatedPct / 100 * this._memoryInfo.getMaxCapacityMb(),
      this._memoryInfo.getMaxCapacityMb(),
    );
    this._predictedBatchSize = Math.min(Math.max(
      getBatchSizeFromUsage(this._memoryInfo.getUsageModelMb(), updatedUsage),
      1,
    ), this._maxBatchSize);

    this._updateAnnotationInBuffer();
    this.notifyChanged();
  }

  updateThroughput(deltaPct, basePct) {
    // Map the delta to a throughput value
    // NOTE: We clamp the values (upper bound for throughput, lower bound for batch size)
    const updatedPct = basePct + deltaPct;
    const updatedThroughput = Math.max(Math.min(
      updatedPct / 100 * this._throughputInfo.getMaxThroughput(),
      this._perfLimits.getThroughputLimit(),
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

    this._updateAnnotationInBuffer();
    this.notifyChanged();
  }

  _updateAnnotationInBuffer() {
    // TODO: Re-implement this function so that it handles
    //       cases where there are multiple text editors.
  }

  _cancelClearView() {
    if (this._clearViewDebounce == null) {
      return;
    }
    clearTimeout(this._clearViewDebounce);
    this._clearViewDebounce = null;
  }

  setClearViewDebounce(clearViewDebounce) {
    this._clearViewDebounce = clearViewDebounce;
  }

  _getAnnotationString() {
    const inputSizeTuple = this._inputInfo.getInputSize().getValuesList();
    const inputSizeCopy = inputSizeTuple.map(x => x);
    if (this._predictedBatchSize != null) {
      inputSizeCopy[0] = Math.round(this._predictedBatchSize);
    }
    return `@innpv size (${inputSizeCopy.join(', ')})`;
  }

  clearPredictions() {
    this._predictedBatchSize = null;
    this._updateAnnotationInBuffer();
    this.notifyChanged();
  }

  getThroughputModel() {
    if (this._throughputInfo == null || this._perfLimits == null) {
      return null;
    }

    if (this._predictedBatchSize == null) {
      return Throughput.fromInfo(this._throughputInfo, this._perfLimits);
    } else {
      return Throughput.fromPrediction(this._throughputInfo, this._perfLimits, this._predictedBatchSize);
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

  getInputInfo() {
    return this._inputInfo;
  }
}

const storeInstance = new BatchSizeStore();

export default storeInstance;
