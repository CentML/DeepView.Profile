'use babel';

export function emptyFor(actionType) {
  return function() {
    return {
      type: actionType,
      payload: {},
    };
  };
}

export function fromPayloadCreator(actionType, payloadCreator) {
  return function(...args) {
    return {
      type: actionType,
      payload: payloadCreator(...args),
    };
  };
}
