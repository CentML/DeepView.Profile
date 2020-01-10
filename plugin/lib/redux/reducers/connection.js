'use babel';

import {
  CONN_INITIATED,
  CONN_INITIALIZED,
  CONN_ERROR,
} from '../actions/types';

import AppState from '../../models/AppState';

export default function(state, action) {
  switch (action.type) {
    case CONN_INITIATED:
      return {
        ...state,
        appState: AppState.CONNECTING,
      };

    default:
      return state;
  }
};
