'use babel';

import AppState from '../../models/AppState';
import PerfVisState from '../../models/PerfVisState';

export default {
  appState: AppState.ACTIVATED,
  perfVisState: PerfVisState.READY,
  errorMessage: '',
  connection: {
    initialized: false,
  },
  memoryBreakdown: null,
  throughput: null,
  memoryUsage: null,
};
