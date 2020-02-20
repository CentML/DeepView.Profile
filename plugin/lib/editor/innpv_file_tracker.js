'use babel';

import FileTracker from './file_tracker';
import {SKYLINE_GUTTER_NAME} from './marker';

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
    for (const editor of this._tracker.editors()) {
      const gutter = editor.gutterWithName(SKYLINE_GUTTER_NAME);
      if (gutter == null) {
        continue;
      }
      gutter.destroy();
    }
    this._tracker.dispose();
    this._tracker = null;
    this._messageSender = null;
    this._store = null;
  }

  _onOpenFilesChange() {
    for (const editor of this._tracker.editors()) {
      if (editor.gutterWithName(SKYLINE_GUTTER_NAME) != null) {
        continue;
      }
      editor.addGutter({
        name: SKYLINE_GUTTER_NAME,
        type: 'decoration',
      });
    }

    this._store.dispatch(ProjectActions.editorsChange({
      editorsByPath: this._tracker.editorsByFilePath(),
    }));
  }

  _onProjectFileSave() {
    // This callback is called each time a project file is saved. We only want
    // to issue another analysis request once all project files have been
    // saved. Therefore we do not take action if there are still modified files
    // in our project, or if an analysis request is already being handled.
    if (this._tracker.hasModifiedFiles() ||
        this._store.getState().perfVisState === PerfVisState.ANALYZING) {
      return;
    }
    this._store.dispatch(AnalysisActions.request());
    this._messageSender.sendAnalysisRequest();
  }

  _onProjectModifiedChange() {
    this._store.dispatch(ProjectActions.modifiedChange({
      modifiedEditorsByFilePath: this._tracker.modifiedEditorsByFilePath(),
    }));
  }
}
