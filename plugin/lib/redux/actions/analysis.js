'use babel';

import path from 'path';
import {Range} from 'atom';

import {emptyFor, fromPayloadCreator} from './utils';
import {
  ANALYSIS_REQ,
  ANALYSIS_REC_BRK,
  ANALYSIS_REC_THPT,
  ANALYSIS_ERROR,
  ANALYSIS_EXPLORE_OP,
  ANALYSIS_EXPLORE_WEIGHT,
  ANALYSIS_EXPLORE_PREV,
  ANALYSIS_EXPLORE_CLEAR,
  ANALYSIS_SET_ACTIVE,
  ANALYSIS_DRAG_THPT,
  ANALYSIS_DRAG_MEM,
  ANALYSIS_PRED_CLEAR,
  ANALYSIS_PRED_CHECKPOINT,
} from './types';

export default {
  request: emptyFor(ANALYSIS_REQ),
  receivedBreakdown: fromPayloadCreator(
    ANALYSIS_REC_BRK,
    ({breakdownResponse}) => ({breakdownResponse}),
  ),
  receivedThroughputAnalysis: fromPayloadCreator(
    ANALYSIS_REC_THPT,
    ({throughputResponse}) => ({throughputResponse}),
  ),
  error: fromPayloadCreator(
    ANALYSIS_ERROR,
    ({errorMessage, errorFileContext}) => ({errorMessage, errorFileContext}),
  ),
  exploreOperation: fromPayloadCreator(
    ANALYSIS_EXPLORE_OP,
    ({newView}) => ({newView}),
  ),
  exploreWeight: fromPayloadCreator(
    ANALYSIS_EXPLORE_WEIGHT,
    ({newView}) => ({newView}),
  ),
  explorePrevious: emptyFor(ANALYSIS_EXPLORE_PREV),
  clearExplored: emptyFor(ANALYSIS_EXPLORE_CLEAR),
  setActive: fromPayloadCreator(
    ANALYSIS_SET_ACTIVE,
    ({currentlyActive}) => ({currentlyActive}),
  ),
  dragThroughput: dragMetricsThunkCreator(ANALYSIS_DRAG_THPT),
  dragMemory: dragMetricsThunkCreator(ANALYSIS_DRAG_MEM),
  clearPredictions: emptyFor(ANALYSIS_PRED_CLEAR),
};

function dragMetricsThunkCreator(actionType) {
  return function({deltaPct, basePct}) {
    return function(dispatch, getState) {
      // 1. Dispatch the original action
      dispatch({type: actionType, payload: {deltaPct, basePct}});

      // 2. Mutate the editor text, if able
      const state = getState();
      const {predictionModels, batchSizeContext, projectRoot} = state;
      if (!predictionModels.batchSizeManipulatable ||
          batchSizeContext == null ||
          projectRoot == null) {
        return;
      }
      if (predictionModels.currentBatchSize == null) {
        return;
      }

      const entryPointPath = path.join(projectRoot, batchSizeContext.filePath);
      atom.workspace.open(
        entryPointPath,
        {initialLine: batchSizeContext.lineNumber - 1},
      )
        .then((editor) => {
          const buffer = editor.getBuffer();
          const range = new Range(
            [batchSizeContext.lineNumber - 1, 0],
            [batchSizeContext.lineNumber - 1, 999],
          );

          let checkpoint = getState().predictionModels.undoCheckpoint;
          if (checkpoint == null) {
            checkpoint = buffer.createCheckpoint();
            dispatch({type: ANALYSIS_PRED_CHECKPOINT, payload: {checkpoint}});
          }

          buffer.setTextInRange(
            range,
            getSkylineBatchSizeSignature(predictionModels.currentBatchSize),
          );
          buffer.groupChangesSinceCheckpoint(checkpoint);
        });
    };
  };
}

function getSkylineBatchSizeSignature(batchSize) {
  return `def skyline_input_provider(batch_size=${batchSize.toFixed(0)}):`;
}
