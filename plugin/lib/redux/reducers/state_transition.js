'use babel';

import PerfVisState from '../../models/PerfVisState';

export default function transitionTo(nextPerfVisState, entireState) {
  if (nextPerfVisState === entireState.perfVisState) {
    // The next state is the same as the current state, so do nothing
    return {};
  }

  switch (entireState.perfVisState) {
    case PerfVisState.SHOWING_PREDICTIONS:
      return fromShowingPredictions(nextPerfVisState, entireState);

    case PerfVisState.ERROR:
      return fromError(nextPerfVisState, entireState);

    case PerfVisState.EXPLORING_WEIGHTS:
    case PerfVisState.EXPLORING_OPERATIONS:
      return fromExploring(nextPerfVisState, entireState);

    default:
      return {perfVisState: nextPerfVisState};
  }
};

function fromShowingPredictions(nextPerfVisState, entireState) {
  return {
    perfVisState: nextPerfVisState,
    predictionModels: {
      ...entireState.predictionModels,
      currentBatchSize: null,
      undoCheckpoint: null,
    },
  };
}

function fromError(nextPerfVisState, entireState) {
  return {
    perfVisState: nextPerfVisState,
    errorMessage: '',
    errorFilePath: null,
    errorLineNumber: null,
  };
}

function fromExploring(nextPerfVisState, entireState) {
  return {
    perfVisState: nextPerfVisState,
    breakdown: {
      ...entireState.breakdown,
      currentView: null,
    },
  };
}
