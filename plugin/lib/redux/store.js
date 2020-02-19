'use babel';

import {createStore, applyMiddleware} from 'redux';
import thunk from 'redux-thunk';
import rootReducer from './reducers/skyline';

// We want to avoid using the singleton pattern. Therefore
// we export a factory function instead.
export default function() {
  return createStore(rootReducer, applyMiddleware(thunk));
}
