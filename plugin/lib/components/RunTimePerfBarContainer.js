'use babel';

import React from 'react';

import PerfBar from './PerfBar';
import PerfBarContainer from './generic/PerfBarContainer';
import OperationInfoStore from '../stores/operationinfo_store';
import PerfVisState from '../models/PerfVisState';

const COLOR_CLASSES = [
  'innpv-bar-color-1',
  'innpv-bar-color-2',
  'innpv-bar-color-3',
  'innpv-bar-color-4',
  'innpv-bar-color-5',
];

export default class RunTimePerfBarContainer extends React.Component {
  constructor(props) {
    super(props);

    const operationInfos = OperationInfoStore.getOperationInfos();
    this.state = {
      operationInfos,
      totalTimeUs: this._getTotalTimeUs(operationInfos),
    };

    this._onStoreChange = this._onStoreChange.bind(this);
    this._perfBarGenerator = this._perfBarGenerator.bind(this);
  }

  componentDidMount() {
    OperationInfoStore.addListener(this._onStoreChange);
  }

  componentWillUnmount() {
    OperationInfoStore.removeListener(this._onStoreChange);
  }

  _onStoreChange() {
    const operationInfos = OperationInfoStore.getOperationInfos();
    this.setState({
      operationInfos,
      totalTimeUs: this._getTotalTimeUs(operationInfos),
    });
  }

  _getTotalTimeUs(operationInfos) {
    return operationInfos.reduce((acc, info) => acc + info.getRuntimeUs(), 0);
  }

  _perfBarGenerator(operationInfo, index, updateMarginTop) {
    return (
      <PerfBar
        key={operationInfo.getBoundName()}
        operationInfo={operationInfo}
        percentage={operationInfo.getRuntimeUs() / this.state.totalTimeUs * 100}
        colorClass={COLOR_CLASSES[index % COLOR_CLASSES.length]}
        updateMarginTop={updateMarginTop}
      />
    );
  }

  render() {
    const {perfVisState} = this.props;
    const disabled = perfVisState === PerfVisState.DEBOUNCING ||
      (perfVisState === PerfVisState.ANALYZING &&
        this.state.operationInfos.length == 0);

    return (
      <PerfBarContainer
        data={this.state.operationInfos}
        perfBarGenerator={this._perfBarGenerator}
        disabled={disabled}
      />
    );
  }
}
