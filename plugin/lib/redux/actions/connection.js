'use babel';

import {generateEmptyActionCreator} from './utils';
import {
  CONN_INITIATED,
  CONN_INITIALIZED,
} from './types';

export default {
  connectionInitiated: generateEmptyActionCreator(CONN_INITIATED),
};
