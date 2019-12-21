'use babel';

import BaseStore from './base_store';
import FileTracker from '../editor/file_tracker';

class ProjectStore extends BaseStore {
  constructor() {
    super();
  }

  reset() {
    this._projectConfig = null;
  }

  receivedProjectConfig(projectConfig) {
    this._projectConfig = projectConfig;
    this.notifyChanged();
  }

  getProjectRoot() {
    return this._projectConfig && this._projectConfig.getProjectRoot();
  }

  getTextEditorsFor(filePath) {
    if (this._projectConfig == null) {
      return [];
    }
    return this._projectConfig.getTextEditorsFor(filePath);
  }
}

const storeInstance = new ProjectStore();

export default storeInstance;
