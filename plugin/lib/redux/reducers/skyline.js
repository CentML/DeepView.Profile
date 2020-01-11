'use babel';

import {
  getActionNamespace,
  isAppAction,
  isConnectionAction,
  isAnalysisAction,
  isProjectAction,
} from '../actions/types';
import analysisReducer from './analysis';
import appReducer from './app';
import connectionReducer from './connection';
import projectReducer from './project';
import initialState from './initial_state';
import Logger from '../../logger';

export default function(state = initialState, action) {
  Logger.debug('Reducer applying action:', action);
  const actionNamespace = getActionNamespace(action);

  if (isAppAction(actionNamespace)) {
    return appReducer(state, action);

  } else if (isConnectionAction(actionNamespace)) {
    return connectionReducer(state, action);

  } else if (isAnalysisAction(actionNamespace)) {
    return analysisReducer(state, action);

  } else if (isProjectAction(actionNamespace)) {
    return projectReducer(state, action);

  } else {
    return state;
  }
};
