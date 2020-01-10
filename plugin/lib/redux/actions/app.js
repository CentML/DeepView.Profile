'use babel';

import {emptyFor} from './utils';
import {
  APP_OPENED,
  APP_CLOSED,
} from './types';

export default {
  appOpened: emptyFor(APP_OPENED),
  appClosed: emptyFor(APP_CLOSED),
};

