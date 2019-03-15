'use babel';

import React from 'react';

import Elastic from './Elastic';
import SourceMarker from '../marker';
import INNPVStore from '../stores/innpv_store';

export default class PerfBar extends React.Component {
  constructor(props) {
    super(props);
    this._op_marker = new SourceMarker(INNPVStore.getEditor());
    this._tooltip = null;
    this._barRef = React.createRef();

    this._handleHoverEnter = this._handleHoverEnter.bind(this);
    this._handleHoverExit = this._handleHoverExit.bind(this);

    this._handleIncrease = this._handleIncrease.bind(this);
    this._handleDecrease = this._handleDecrease.bind(this);
    this._handleRestore = this._handleRestore.bind(this);
  }

  componentDidMount() {
    this._op_marker.register(this.props.operationInfo.getLocation());
    this._registerTooltip();
  }

  componentDidUpdate(prevProps) {
    this._op_marker.reconcileLocation(
      prevProps.operationInfo.getLocation(),
      this.props.operationInfo.getLocation(),
    );
    this._updateTooltip(prevProps);
  }

  componentWillUnmount() {
    this._op_marker.remove();
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

  _handleHoverEnter() {
    this._op_marker.showDecoration({type: 'line', class: 'innpv-line-highlight'});
  }

  _handleHoverExit() {
    this._op_marker.hideDecoration();
  }

  _handleIncrease() {
  }

  _handleDecrease() {
  }

  _handleRestore() {
  }

  render() {
    const {operationInfo} = this.props;
    const resizable = operationInfo.getHintsList().length !== 0

    return (
      <Elastic
        className={`innpv-perfbar-wrap ${resizable ? "innpv-perfbar-resizable" : ""}`}
        disabled={!resizable}
        heightPct={this.props.percentage}
        updateMarginTop={this.props.updateMarginTop}
        handleShrink={this._handleDecrease}
        handleGrow={this._handleIncrease}
        handleSnapBack={this._handleRestore}
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
