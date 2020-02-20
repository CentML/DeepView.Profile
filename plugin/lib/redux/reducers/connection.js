'use babel';

import AppState from '../../models/AppState';
import PerfVisState from '../../models/PerfVisState';
import {
  CONN_CONNECTING,
  CONN_INITIALIZING,
  CONN_INITIALIZED,
  CONN_ERROR,
  CONN_LOST,
  CONN_INCR_SEQ,
} from '../actions/types';
import transitionTo from './state_transition';
import initialState from './initial_state';

export default function(state, action) {
  switch (action.type) {
    case CONN_CONNECTING:
      return {
        ...state,
        appState: AppState.CONNECTING,
      };

    case CONN_INITIALIZING:
      return {
        ...state,
        connection: {
          ...state.connection,
          onTimeout: action.payload.onTimeout,
        },
      };

    case CONN_INITIALIZED:
      return {
        ...state,
        appState: AppState.CONNECTED,
        ...transitionTo(PerfVisState.READY, state),
        errorMessage: '',
        connection: {
          ...state.connection,
          initialized: true,
          onTimeout: null,
        },
        projectRoot: action.payload.projectRoot,
      };

    case CONN_ERROR:
    case CONN_LOST:
      return {
        ...initialState,
        appState: AppState.OPENED,
        errorMessage: action.payload.errorMessage,
      };

    case CONN_INCR_SEQ:
      return {
        ...state,
        connection: {
          ...state.connection,
          sequenceNumber: state.connection.sequenceNumber + 1,
        },
      };

    default:
      return state;
  }
};
