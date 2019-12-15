'use babel';

import BaseStore from './base_store';

import MemoryUsage from '../models/MemoryUsage';

class AnalysisStore extends BaseStore {
  constructor() {
    super();
  }

  reset() {
    this._memoryBreakdown = null;
    this._overallMemoryUsage = null;
  }

  receivedMemoryUsage(memoryUsageResponse) {
    this._overallMemoryUsage = MemoryUsage.fromMemoryUsageResponse(memoryUsageResponse);
    // TODO: Construct a model for the memory breakdown
    this.notifyChanged();
  }

  getMemoryBreakdown() {
    // This data is used by the MemoryPerfBarContainer
    return this._memoryBreakdown;
  }

  getOverallMemoryUsage() {
    // This data is used by the Memory component
    return this._overallMemoryUsage;
  }
}

const storeInstance = new AnalysisStore();

export default storeInstance;
