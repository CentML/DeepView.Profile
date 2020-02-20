'use babel';

import {fromPayloadCreator} from './utils';
import {
  PROJECT_MODIFIED_CHANGE,
  PROJECT_EDITORS_CHANGE,
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
};

