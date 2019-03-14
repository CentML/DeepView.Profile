'use babel';

export default class SourceMarker {
  constructor(editor) {
    this._editor = editor;
    this._marker = null;
    this._decoration = null;
  }

  register(location) {
    if (location == null) {
      return;
    }
    this._marker = this._editor.markBufferPosition(
      [location.getLine(), location.getColumn()],
    );
  }

  reconcileLocation(prevLocation, newLocation) {
    if (prevLocation != null && newLocation != null &&
        newLocation.getLine() === prevLocation.getLine() &&
        newLocation.getColumn() === prevLocation.getColumn()) {
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
