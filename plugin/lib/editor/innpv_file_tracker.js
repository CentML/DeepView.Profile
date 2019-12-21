'use babel';

import FileTracker from './file_tracker';

import INNPVStore from '../stores/innpv_store';
import AnalysisStore from '../stores/analysis_store';
import ProjectStore from '../stores/project_store';
import PerfVisState from '../models/PerfVisState';

// The purpose of the INNPVFileTracker is to bind to the stores
// that we use in INNPV. This allows the FileTracker to remain
// generic (i.e. no dependence on INNPV).

export default class INNPVFileTracker {
  constructor(projectRoot, messageSender) {
    this._messageSender = messageSender;
    this._tracker = new FileTracker({
      projectRoot,
      onOpenFilesChange: this._onOpenFilesChange.bind(this),
      onProjectFileSave: this._onProjectFileSave.bind(this),
      onProjectModifiedChange: this._onProjectModifiedChange.bind(this),
    });

    this._projectConfig = {
      getTextEditorsFor: this._tracker.getTextEditorsFor.bind(this._tracker),
      getProjectRoot: () => projectRoot,
    };
    ProjectStore.receivedProjectConfig(this._projectConfig);
  }

  dispose() {
    this._tracker.dispose();
    this._tracker = null;
    this._messageSender = null;
    this._projectConfig = null;
  }

  _onOpenFilesChange() {
    ProjectStore.receivedProjectConfig(this._projectConfig);
  }

  _onProjectFileSave() {
    if (this._tracker.isProjectModified() ||
        INNPVStore.getPerfVisState() !== PerfVisState.READY) {
      return;
    }
    INNPVStore.setPerfVisState(PerfVisState.ANALYZING);
    this._messageSender.sendAnalysisRequest();
  }

  _onProjectModifiedChange() {
    const modified = this._tracker.isProjectModified();
    const perfVisState = INNPVStore.getPerfVisState();

    if (modified && perfVisState === PerfVisState.READY) {
      INNPVStore.setPerfVisState(PerfVisState.MODIFIED);

    } else if (!modified && perfVisState === PerfVisState.MODIFIED) {
      INNPVStore.setPerfVisState(PerfVisState.READY);
    }
  }
}
