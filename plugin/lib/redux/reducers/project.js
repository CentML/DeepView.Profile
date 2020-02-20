'use babel';

import {
  PROJECT_MODIFIED_CHANGE,
  PROJECT_EDITORS_CHANGE,
} from '../actions/types';
import PerfVisState from '../../models/PerfVisState';

export default function(state, action) {
  switch (action.type) {
    case PROJECT_MODIFIED_CHANGE: {
      const {modifiedEditorsByFilePath} = action.payload;
      return {
        ...state,
        projectModified: modifiedEditorsByFilePath.size > 0,
      };
    }

    case PROJECT_EDITORS_CHANGE:
      return {
        ...state,
        editorsByPath: action.payload.editorsByPath,
      };

    default:
      return state;
  }
};
