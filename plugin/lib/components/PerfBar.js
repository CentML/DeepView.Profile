'use babel';

import React from 'react';

import Elastic from './Elastic';

export default class PerfBar extends React.Component {
  constructor(props) {
    super(props);
    this._marker = null;
    this._decoration = null;
    this._tooltip = null;
    this._barRef = React.createRef();

    this._mouseDown = false;

    this._handleHoverEnter = this._handleHoverEnter.bind(this);
    this._handleHoverExit = this._handleHoverExit.bind(this);
  }

  componentDidMount() {
    this._registerCodeMarker();
    this._registerTooltip();
  }

  componentDidUpdate(prevProps) {
    this._updateMarker(prevProps);
    this._updateTooltip(prevProps);
  }

  componentWillUnmount() {
    this._clearCodeMarker();
    this._clearTooltip();
  }

  _updateTooltip(prevProps) {
    const operationInfo = this.props.operationInfo;
    const prevOperationInfo = prevProps.operationInfo;
    if (operationInfo.getOpName() === prevOperationInfo.getOpName() &&
        operationInfo.getRuntimeUs() === prevOperationInfo.getRuntimeUs() &&
        this.props.percentage === prevProps.percentage) {
      return;
    }
    this._clearTooltip();
    this._registerTooltip();
  }

  _registerTooltip() {
    this._tooltip = atom.tooltips.add(
      this._barRef.current,
      {
        title: this._generateTooltipHTML(),
        placement: 'right',
        html: true,
      },
    )
  }

  _clearTooltip() {
    if (this._tooltip == null) {
      return;
    }
    this._tooltip.dispose();
    this._tooltip = null;
  }

  _generateTooltipHTML() {
    const {operationInfo, percentage} = this.props;
    return `<strong>${operationInfo.getOpName()}</strong><br/>` +
      `Run Time: ${operationInfo.getRuntimeUs().toFixed(2)} us<br/>` +
      `Weight: ${percentage.toFixed(2)}%`;
  }

  _updateMarker(prevProps) {
    const operationInfo = this.props.operationInfo;
    const prevOperationInfo = prevProps.operationInfo;
    if (operationInfo.getLine() === prevOperationInfo.getLine() &&
        operationInfo.getColumn() === prevOperationInfo.getColumn()) {
      return;
    }
    this._clearCodeMarker();
    this._registerCodeMarker();
  }

  _registerCodeMarker() {
    const {editor, operationInfo} = this.props;
    // Line & Column are 1-based indices whereas Atom wants 0-based indices
    this._marker = editor.markBufferPosition(
      [operationInfo.getLine() - 1, operationInfo.getColumn() - 1],
    );
  }

  _clearCodeMarker() {
    if (this._marker == null) {
      return;
    }
    this._marker.destroy();
    this._marker = null;
  }

  _handleHoverEnter() {
    this._decoration = this.props.editor.decorateMarker(
      this._marker,
      {type: 'line', class: 'innpv-line-highlight'},
    );
  }

  _handleHoverExit() {
    if (this._decoration == null) {
      return;
    }
    this._decoration.destroy();
    this._decoration = null;
  }

  render() {
    return (
      <Elastic
        className="innpv-perfbar-wrap"
        heightPct={this.props.percentage}
        updateMarginTop={this.props.updateMarginTop}
      >
        <div
          ref={this._barRef}
          className={`innpv-perfbar ${this.props.colorClass}`}
          onMouseEnter={this._handleHoverEnter}
          onMouseLeave={this._handleHoverExit}
        />
      </Elastic>
    );
  }
}
