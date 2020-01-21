'use babel';

import {fromPayloadCreator} from './utils';
import {
  CONFIG_CHANGED,
} from './types';

export default {
  // The payload contains the new value of the configuration setting
  // (the key refers to the configuration key).
  configChanged: fromPayloadCreator(CONFIG_CHANGED, (payload) => payload),
};

