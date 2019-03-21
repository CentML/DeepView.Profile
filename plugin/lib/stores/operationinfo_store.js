'use babel';

import BaseStore from './base_store';

class OperationInfoStore extends BaseStore {
  constructor() {
    super();
  }

  reset() {
    this._operationInfos = [];
  }

  getOperationInfos() {
    return this._operationInfos;
  }

  setOperationInfos(operationInfos) {
    this._operationInfos = operationInfos;
    this.notifyChanged();
  }
}

const storeInstance = new OperationInfoStore();

export default storeInstance;
