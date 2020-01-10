'use babel';

import {
  isAppAction,
  isConnectionAction,
  isAnalysisAction,
} from '../actions/types';
import analysisReducer from './analysis';
import appReducer from './app';
import connectionReducer from './connection';
import initialState from './initial_state';
import Logger from '../../logger';

export default function(state = initialState, action) {
  Logger.debug('Reducer applying action:', action);

  if (isAppAction(action)) {
    return appReducer(state, action);

  } else if (isConnectionAction(action)) {
    return connectionReducer(state, action);

  } else if (isAnalysisAction(action)) {
    return analysisReducer(state, action);

  } else {
    return state;
  }
};
