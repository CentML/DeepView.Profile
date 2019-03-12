'use babel';

export default class SourceMarker {
  constructor(editor) {
    this._editor = editor;
    this._marker = null;
    this._decoration = null;
  }

  register(location) {
    // Line & Column are 1-based indices whereas Atom wants 0-based indices
    this._marker = this._editor.markBufferPosition(
      [location.getLine() - 1, location.getColumn() - 1],
    );
  }

  reconcileLocation(prevLocation, newLocation) {
    if (newLocation.getLine() === prevLocation.getLine() &&
        newLocation.getColumn() === prevLocation.getColumn()) {
      return;
    }
    this._removeMarker();
    this._registerMarker(newLocation);
  }

  remove() {
    if (this._marker == null) {
      return;
    }
    this._marker.destroy();
    this._marker = null;
  }

  showDecoration(options) {
    if (this._marker == null) {
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
