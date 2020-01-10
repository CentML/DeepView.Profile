'use babel';

import {createStore} from 'redux';
import rootReducer from './reducers/skyline';

export default createStore(rootReducer);
