'use babel';

import {generateEmptyActionCreator} from './utils';
import {
  APP_OPENED,
  APP_CLOSED,
} from './types';

export default {
  appOpened: generateEmptyActionCreator(APP_OPENED),
  appClosed: generateEmptyActionCreator(APP_CLOSED),
};

