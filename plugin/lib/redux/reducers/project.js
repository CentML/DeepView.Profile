'use babel';

import {
  PROJECT_MODIFIED_CHANGE,
  PROJECT_EDITORS_CHANGE,
  PROJECT_CAN_PROFILE,
} from '../actions/types';
import transitionTo from './state_transition';
import PerfVisState from '../../models/PerfVisState';

export default function(state, action) {
  switch (action.type) {
    case PROJECT_MODIFIED_CHANGE: {
      const {modifiedEditorsByFilePath} = action.payload;

      // By default, we consider the project to have modifications if there
      // exists one project editor with changes.
      let projectModified = modifiedEditorsByFilePath.size > 0;

      // However when predictions occur, we mutate the project. We don't want
      // to mark the project modified in this case.
      const {batchSizeContext, perfVisState} = state;
      if (perfVisState === PerfVisState.SHOWING_PREDICTIONS &&
          batchSizeContext != null &&
          modifiedEditorsByFilePath.size === 1 &&
          modifiedEditorsByFilePath.has(batchSizeContext.filePath)) {
        projectModified = false;
      }

      return {
        ...state,
        projectModified,
        modifiedEditorsByPath: modifiedEditorsByFilePath,
      };
    }

    case PROJECT_EDITORS_CHANGE:
      return {
        ...state,
        editorsByPath: action.payload.editorsByPath,
      };

    case PROJECT_CAN_PROFILE:
      return {
        ...state,
        ...transitionTo(PerfVisState.READY_STALE, state),
      };

    default:
      return state;
  }
};
