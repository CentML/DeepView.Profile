'use babel';

import React from 'react'

import SourceMarker from '../editor/marker';
import INNPVStore from '../stores/innpv_store';

class UsageHighlight extends React.Component {
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

    if (!prevProps.show && this.props.show) {
      this._showDecoration();
    } else if (prevProps.show && !this.props.show) {
      this._hideDecoration();
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
      this._showDecoration();
    }
  }

  _showDecoration() {
    this._marker.showDecoration({type: 'line', class: 'innpv-line-highlight'});
  }

  _hideDecoration() {
    this._marker.hideDecoration();
  }

  render() {
    return null;
  }
}

UsageHighlight.defaultProps = {
  show: false,
  column: 1,
};

export default UsageHighlight;
