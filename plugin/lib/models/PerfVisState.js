'use babel';

export default {
  // When the plugin is waiting for results from the profiler
  ANALYZING: 'analyzing',

  // When the plugin is ready for interactions
  READY: 'ready',

  // When there are no modified files but Skyline could be showing stale performance data
  // NOTE: This state only occurs if the user disables automatic profiling.
  READY_STALE: 'ready_stale',

  // When the plugin is showing the user's predictions
  SHOWING_PREDICTIONS: 'showing_predictions',

  // When there has been an error processing the user's input
  ERROR: 'error',

  // When the user has double clicked the breakdown and is exploring parts of the tree
  EXPLORING_WEIGHTS: 'exploring_weights',
  EXPLORING_OPERATIONS: 'exploring_operations',
};
