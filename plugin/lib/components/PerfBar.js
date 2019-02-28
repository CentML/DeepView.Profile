'use babel';

import React from 'react';

export default class PerfBar extends React.Component {
  constructor(props) {
    super(props);
    this._marker = null;
    this._decoration = null;

    this._handleHoverEnter = this._handleHoverEnter.bind(this);
    this._handleHoverExit = this._handleHoverExit.bind(this);
  }

  componentDidMount() {
    this._registerCodeMarker();
  }

  componentDidUpdate(prevProps) {
    const operationInfo = this.props.operationInfo;
    const prevOperationInfo = prevProps.operationInfo;
    if (operationInfo.getLine() === prevOperationInfo.getLine() &&
        operationInfo.getColumn() === prevOperationInfo.getColumn()) {
      return;
    }
    this._clearCodeMarker();
    this._registerCodeMarker();
  }

  componentWillUnmount() {
    this._clearCodeMarker();
  }

  _registerCodeMarker() {
    const {editor, operationInfo} = this.props;
    // Line & Column are 1-based indices whereas Atom wants 0-based indices
    this._marker = editor.markBufferPosition(
      [operationInfo.getLine() - 1, operationInfo.getColumn() - 1],
    );
  }

  _clearCodeMarker() {
    this._marker.destroy();
  }

  _handleHoverEnter() {
    this._decoration = this.props.editor.decorateMarker(
      this._marker,
      {type: 'line', class: 'innpv-line-highlight'},
    );
  }

  _handleHoverExit() {
    this._decoration.destroy();
    this._decoration = null;
  }

  render() {
    return (
      <div
        className={`innpv-perfbar ${this.props.colorClass}`}
        style={{height: `${this.props.percentage}%`}}
        onMouseEnter={this._handleHoverEnter}
        onMouseLeave={this._handleHoverExit}
      />
    );
  }
}
