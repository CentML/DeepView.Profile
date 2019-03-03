'use babel';

export default {
  // When the plugin is waiting for results from the profiler
  ANALYZING: 'analyzing',

  // When the plugin is ready for interactions
  READY: 'ready',

  // When the plugin has detected changes, but is still waiting
  // for the user to stop typing
  DEBOUNCING: 'debouncing',

  // When the plugin is showing the user's predictions
  SHOWING_PREDICTIONS: 'showing_predictions',

  // When there has been an error processing the user's input
  ERROR: 'error',
};
