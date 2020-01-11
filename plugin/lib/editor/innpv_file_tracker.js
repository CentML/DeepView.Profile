'use babel';

import FileTracker from './file_tracker';

import PerfVisState from '../models/PerfVisState';
import Logger from '../logger';
import AnalysisActions from '../redux/actions/analysis';
import ProjectActions from '../redux/actions/project';

// The purpose of the INNPVFileTracker is to bind to the store
// that we use in Skyline. This allows the FileTracker to remain
// generic (i.e. no dependence on Skyline).

export default class INNPVFileTracker {
  constructor(projectRoot, messageSender, store) {
    this._messageSender = messageSender;
    this._store = store;
    this._tracker = new FileTracker({
      projectRoot,
      onOpenFilesChange: this._onOpenFilesChange.bind(this),
      onProjectFileSave: this._onProjectFileSave.bind(this),
      onProjectModifiedChange: this._onProjectModifiedChange.bind(this),
    });
  }

  dispose() {
    this._tracker.dispose();
    this._tracker = null;
    this._messageSender = null;
    this._store = null;
  }

  _onOpenFilesChange() {
    this._store.dispatch(ProjectActions.editorsChange({
      editorsByPath: this._tracker.editorsByFilePath(),
    }));
  }

  _onProjectFileSave() {
    if (this._tracker.isProjectModified() ||
        this._store.getState().perfVisState === PerfVisState.ANALYZING) {
      return;
    }
    this._store.dispatch(AnalysisActions.request());
    this._messageSender.sendAnalysisRequest();
  }

  _onProjectModifiedChange() {
    this._store.dispatch(ProjectActions.modifiedChange({
      modified: this._tracker.isProjectModified()
    }));
  }
}
