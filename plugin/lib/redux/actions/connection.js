'use babel';

import {emptyFor, fromPayloadCreator} from './utils';
import {
  CONN_CONNECTING,
  CONN_INITIALIZING,
  CONN_INITIALIZED,
  CONN_ERROR,
  CONN_LOST,
  CONN_INCR_SEQ,
} from './types';

export default {
  connecting: emptyFor(CONN_CONNECTING),
  initializing: fromPayloadCreator(CONN_INITIALIZING, ({onTimeout}) => ({onTimeout})),
  initialized: fromPayloadCreator(CONN_INITIALIZED, ({projectRoot}) => ({projectRoot})),
  error: fromPayloadCreator(CONN_ERROR, ({errorMessage}) => ({errorMessage})),
  lost: fromPayloadCreator(CONN_LOST, ({errorMessage}) => ({errorMessage})),
  incrementSequence: emptyFor(CONN_INCR_SEQ),
};
