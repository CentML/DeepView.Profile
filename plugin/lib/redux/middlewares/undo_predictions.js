'use babel';

// This middleware is used to revert any changes to the code that we made while
// the user dragged the key performance metrics. We do this by detecting the
// presence and then removal of an Atom undo checkpoint. When this happens, we
// make a best-effort attempt to revert any changes made.

export default function undoPredictions({getState}) {
  return next => action => {
    const prevUndoCheckpoint = getState().predictionModels.undoCheckpoint;
    const returnValue = next(action);
    const state = getState();

    if (prevUndoCheckpoint == null ||
        state.predictionModels.undoCheckpoint != null) {
      // We only care about the case where the checkpoint goes from not null to
      // null.
      return returnValue;
    }

    const {batchSizeContext, editorsByPath} = state;
    if (batchSizeContext == null ||
        !editorsByPath.has(batchSizeContext.filePath)) {
      // Handle the case where the editor may have been closed.
      return returnValue;
    }

    // We only consider one editor since, if there are multiple editors open,
    // they will all share the same underlying text buffer.
    const editors = editorsByPath.get(batchSizeContext.filePath);
    editors[0].getBuffer().revertToCheckpoint(prevUndoCheckpoint);
    return returnValue;
  };
};
