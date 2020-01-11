'use babel';

import AppState from '../../models/AppState';
import PerfVisState from '../../models/PerfVisState';

export default {
  // Global application states
  appState: AppState.ACTIVATED,
  perfVisState: PerfVisState.READY,
  errorMessage: '',

  // Server connection state
  connection: {
    initialized: false,
    sequenceNumber: 0,
    onTimeout: null,
  },

  // Project state
  projectRoot: null,
  editorsByPath: new Map(),

  // Analysis received
  memoryBreakdown: null,
  memoryUsage: null,
  throughput: null,
};
