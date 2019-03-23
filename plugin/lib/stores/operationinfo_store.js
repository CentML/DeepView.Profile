'use babel';

import BaseStore from './base_store';

class OperationInfoStore extends BaseStore {
  constructor() {
    super();
  }

  reset() {
    this._operationInfos = [];
    this._clearViewDebounce = null;
  }

  getOperationInfos() {
    return this._operationInfos;
  }

  setOperationInfos(operationInfos) {
    // Prevent the view from being cleared if we receive
    // results really quickly.
    if (this._clearViewDebounce != null) {
      clearTimeout(this._clearViewDebounce);
      this._clearViewDebounce = null;
    }

    this._operationInfos = operationInfos;
    this.notifyChanged();
  }

  setClearViewDebounce(clearViewDebounce) {
    this._clearViewDebounce = clearViewDebounce;
  }
}

const storeInstance = new OperationInfoStore();

export default storeInstance;
