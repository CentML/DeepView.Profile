'use babel';

import {APP_OPENED, APP_CLOSED} from '../actions/types';
import AppState from '../../models/AppState';
import initialState from './initial_state';

export default function(state, action) {
  switch (action.type) {
    case APP_OPENED:
      return {
        ...state,
        appState: AppState.OPENED,
      };

    case APP_CLOSED:
      return initialState;

    default:
      return state;
  }
};
