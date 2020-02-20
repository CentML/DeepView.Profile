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
  projectModified: false,

  // Analysis received
  memoryBreakdown: null,
  memoryUsage: null,
  throughput: null,
  runTimeBreakdown: null,
  batchSize: null,
  batchSizeContext: null,

  // State associated with the breakdowns
  // (display and navigation)
  breakdown: {
    operationTree: null,
    weightTree: null,

    currentView: null,
    currentlyActive: null,
  },
  runTime: {
    trackedMs: null,
    otherMs: null,
    otherNode: null,
  },
  memory: {
    trackedBytes: null,
    untrackedBytes: null,
    untrackedNode: null,
  },
  peakUsageBytes: null,
  memoryCapacityBytes: null,
  iterationRunTimeMs: null,

  // Linear models that enable the interactive views
  predictionModels: {
    runTimeMs: null,
    peakUsageBytes: null,
    currentBatchSize: null,
    maxBatchSize: null,
    batchSizeManipulatable: false,
  },
};
