'use babel';

import BaseStore from './base_store';
import FileTracker from '../file_tracker';

class ProjectStore extends BaseStore {
  constructor() {
    super();
    this._onOpenFilesChange = this._onOpenFilesChange.bind(this);
  }

  reset() {
    this._projectRoot = null;
    this._entryPoint = null;
    this._resetFileTracker();
  }

  receivedProjectConfig(projectRoot, entryPoint) {
    this._projectRoot = projectRoot;
    this._entryPoint = entryPoint;
    this._resetFileTracker();
    this._fileTracker = new FileTracker({
      projectRoot,
      onOpenFilesChange: this._onOpenFilesChange,
    });
    this.notifyChanged();
  }

  getProjectRoot() {
    return this._projectRoot;
  }

  getTextEditorsFor(filePath) {
    if (this._fileTracker == null) {
      return [];
    }
    return this._fileTracker.getTextEditorsFor(filePath);
  }

  _resetFileTracker() {
    if (this._fileTracker == null) {
      return;
    }
    this._fileTracker.dispose();
    this._fileTracker = null;
  }

  _onOpenFilesChange() {
    this.notifyChanged();
  }
}

const storeInstance = new ProjectStore();

export default storeInstance;
