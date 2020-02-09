'use babel';

import React from 'react'

import {SourceMarker} from '../../editor/marker';

class InlineHighlight extends React.Component {
  componentDidMount() {
    this._setUpMarker();
  }

  componentDidUpdate(prevProps) {
    const {editor} = this.props;
    const prevEditor = prevProps.editor;
    if (prevEditor !== editor) {
      this._marker.remove();
      this._setUpMarker();
      return;
    }

    const {lineNumber, column} = this.props;
    const prevLineNumber = prevProps.lineNumber;
    const prevColumn = prevProps.column;
    this._marker.reconcileLocation(
      {lineNumber: prevLineNumber, column: prevColumn},
      {lineNumber, column},
    );

    if (this.props.show) {
      this._showDecorations();
    } else {
      this._hideDecorations();
    }
  }

  componentWillUnmount() {
    this._marker.remove();
  }

  _setUpMarker() {
    const {editor, lineNumber, column, show} = this.props;
    this._marker = new SourceMarker(editor);
    this._marker.register({lineNumber, column});

    if (show) {
      this._showDecorations();
    }
  }

  _showDecorations() {
    this._marker.showDecorations(this.props.decorations);
  }

  _hideDecorations() {
    this._marker.hideDecorations();
  }

  render() {
    return null;
  }
}

InlineHighlight.defaultProps = {
  show: false,
  column: 1,
  decorations: [],
};

export default InlineHighlight;
