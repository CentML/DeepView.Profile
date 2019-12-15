'use babel';

import BaseStore from './base_store';

import MemoryBreakdown from '../models/MemoryBreakdown';
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
    this._memoryBreakdown = MemoryBreakdown.fromMemoryUsageResponse(memoryUsageResponse);
    this._overallMemoryUsage = MemoryUsage.fromMemoryUsageResponse(memoryUsageResponse);
    this.notifyChanged();
  }

  getMemoryBreakdown() {
    // This data is used by the MemoryBreakdown component
    return this._memoryBreakdown;
  }

  getOverallMemoryUsage() {
    // This data is used by the Memory component
    return this._overallMemoryUsage;
  }
}

const storeInstance = new AnalysisStore();

export default storeInstance;
