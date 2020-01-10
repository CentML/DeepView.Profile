'use babel';

import {createStore} from 'redux';
import rootReducer from './reducers/skyline';

// We want to avoid using the singleton pattern. Therefore
// we export a factory function instead.
export default function() {
  return createStore(rootReducer);
}
