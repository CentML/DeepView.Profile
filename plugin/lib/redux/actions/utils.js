'use babel';

export function generateEmptyActionCreator(actionType) {
  return function() {
    return {
      type: actionType,
      payload: {},
    };
  };
}

export function generateActionCreatorFromPayloadCreator(actionType, payloadCreator) {
  return function(...args) {
    return {
      type: actionType,
      payload: payloadCreator(...args),
    };
  };
}
