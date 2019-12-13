'use babel';

import React from 'react';

import PerfBar from './generic/PerfBar';
import PerfHint from './PerfHint';
import UsageHighlight from './UsageHighlight';
import SourceMarker from '../marker';
import PerfHintState from '../models/PerfHintState';
import INNPVStore from '../stores/innpv_store';

export default class RunTimePerfBar extends React.Component {
  constructor(props) {
    super(props);

    this._renderPerfHints = this._renderPerfHints.bind(this);
  }

  _generateTooltipHTML() {
    const {operationInfo, percentage} = this.props;
    return `<strong>${operationInfo.getOpName()}</strong> (${operationInfo.getBoundName()})<br/>` +
      `Run Time: ${operationInfo.getRuntimeUs().toFixed(2)} us<br/>` +
      `Weight: ${percentage.toFixed(2)}%`;
  }

  _renderPerfHints(isActive, perfHintState) {
    const {operationInfo} = this.props;

    return [
      ...operationInfo.getHintsList().map(perfHint =>
        <PerfHint perfHint={perfHint} perfHintState={perfHintState} />
      ),
      ...operationInfo.getUsagesList().map(location =>
        <UsageHighlight location={location} show={isActive} />
      ),
    ];
  }

  render() {
    const {operationInfo, ...rest} = this.props;
    const resizable = operationInfo.getHintsList().length !== 0

    return (
      <PerfBar
        resizeable={resizeable}
        renderPerfHints={this._renderPerfHints}
        tooltipHTML={this._generateTooltipHTML()}
        {...rest}
      />
    );
  }
}
