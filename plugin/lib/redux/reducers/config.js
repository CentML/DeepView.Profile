'use babel';

import {
  getActionNamespace,
  isConfigAction,
} from '../actions/types';

import configSchema from '../../config/schema';

const defaultConfig = (() => {
  const config = {};
  for (const key in configSchema) {
    if (!Object.prototype.hasOwnProperty.call(configSchema, key)) {
      continue;
    }
    config[key] = configSchema[key].default;
  }
  return config;
})();

export default function(config = defaultConfig, action) {
  if (!isConfigAction(getActionNamespace(action))) {
    return config;
  }
  return {
    ...config,
    ...action.payload,
  };
}
