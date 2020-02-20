'use babel';

import {
  PROJECT_MODIFIED_CHANGE,
  PROJECT_EDITORS_CHANGE,
} from '../actions/types';
import transitionTo from './state_transition';
import PerfVisState from '../../models/PerfVisState';
import Logger from '../../logger';

export default function(state, action) {
  switch (action.type) {
    case PROJECT_MODIFIED_CHANGE: {
      const {perfVisState} = state;
      const {modified} = action.payload;
      if (modified && perfVisState !== PerfVisState.MODIFIED) {
        return {
          ...state,
          ...transitionTo(PerfVisState.MODIFIED, state),
        };

      } else if (!modified && perfVisState === PerfVisState.MODIFIED) {
        return {
          ...state,
          ...transitionTo(PerfVisState.READY, state),
        };
      } else {
        Logger.warn(
          'Modified change unexpected case. Modified:',
          modified,
          'PerfVisState:',
          perfVisState,
        );
        return state;
      }
    }

    case PROJECT_EDITORS_CHANGE:
      return {
        ...state,
        editorsByPath: action.payload.editorsByPath,
      };

    default:
      return state;
  }
};
