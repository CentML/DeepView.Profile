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
import configReducer from './config';
import connectionReducer from './connection';
import projectReducer from './project';
import initialState from './initial_state';

import Logger from '../../logger';

function skylineReducer(state = initialState, action) {
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

export default function(state, action) {
  // We don't use combineReducers() here to avoid placing all the Skyline state
  // under another object. This way the config remains a "top level" state
  // object alongside other Skyline state properties.
  Logger.debug('Reducer applying action:', action);
  if (state === undefined) {
    // Initial state
    const newState = skylineReducer(undefined, action);
    newState.config = configReducer(undefined, action);
    return newState;
  }

  const {config, ...rest} = state;
  const newState = skylineReducer(rest, action);
  newState.config = configReducer(config, action);
  return newState;
};
