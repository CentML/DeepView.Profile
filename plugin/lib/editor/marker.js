'use babel';

import Logger from '../logger';

export const SKYLINE_GUTTER_NAME = 'skyline';

export class SourceMarker {
  constructor(editor) {
    this._editor = editor;
    this._marker = null;
    this._decorations = [];
  }

  register({lineNumber, column}) {
    // NOTE: INNPV uses 1-based line numbers and columns, but Atom uses
    // 0-based line numbers and columns.
    this._marker = this._editor.markBufferPosition([lineNumber - 1, column - 1]);
  }

  reconcileLocation(prevLocation, newLocation) {
    const prevLineNumber = prevLocation.lineNumber;
    const prevColumn = prevLocation.column;
    const {lineNumber, column} = newLocation;

    if (prevLocation != null && newLocation != null &&
        lineNumber === prevLineNumber &&
        column === prevColumn) {
      return;
    }
    this.remove();
    this.register(newLocation);
  }

  remove() {
    if (this._marker == null) {
      return;
    }
    this._marker.destroy();
    this._marker = null;
    this._decorations = [];
  }

  showDecorations(decorationOptions) {
    if (this._marker == null || this._decorations.length !== 0) {
      return;
    }
    for (const options of decorationOptions) {
      if (options.type === 'gutter') {
        const gutter = this._editor.gutterWithName(SKYLINE_GUTTER_NAME);
        if (gutter == null) {
          Logger.warn('Missing Skyline gutter in tracked editor.');
          continue;
        }
        this._decorations.push(
          gutter.decorateMarker(this._marker, options),
        );

      } else {
        this._decorations.push(
          this._editor.decorateMarker(this._marker, options),
        );
      }
    }
  }

  hideDecorations() {
    if (this._decorations.length === 0) {
      return;
    }
    for (const decoration of this._decorations) {
      decoration.destroy();
    }
    this._decorations = [];
  }
};
