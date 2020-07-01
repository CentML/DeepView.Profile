'use babel';

import {emptyFor, fromPayloadCreator} from './utils';
import {
  PROJECT_MODIFIED_CHANGE,
  PROJECT_EDITORS_CHANGE,
  PROJECT_CAN_PROFILE,
} from './types';

export default {
  modifiedChange: fromPayloadCreator(
    PROJECT_MODIFIED_CHANGE,
    ({modifiedEditorsByFilePath}) => ({modifiedEditorsByFilePath}),
  ),
  editorsChange: fromPayloadCreator(
    PROJECT_EDITORS_CHANGE,
    ({editorsByPath}) => ({editorsByPath}),
  ),
  canProfile: emptyFor(PROJECT_CAN_PROFILE),
};

