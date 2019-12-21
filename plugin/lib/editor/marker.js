'use babel';

export default class SourceMarker {
  constructor(editor) {
    this._editor = editor;
    this._marker = null;
    this._decoration = null;
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
  }

  showDecoration(options) {
    if (this._marker == null || this._decoration != null) {
      return;
    }
    this._decoration = this._editor.decorateMarker(this._marker, options);
  }

  hideDecoration() {
    if (this._decoration == null) {
      return;
    }
    this._decoration.destroy();
    this._decoration = null;
  }
}
