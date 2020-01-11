'use babel';

import AppState from '../../models/AppState';
import PerfVisState from '../../models/PerfVisState';

export default {
  appState: AppState.ACTIVATED,
  perfVisState: PerfVisState.READY,
  errorMessage: '',
  connection: {
    initialized: false,
    sequenceNumber: 0,
    onTimeout: null,
  },
  memoryBreakdown: null,
  throughput: null,
  memoryUsage: null,
  projectRoot: null,
};
