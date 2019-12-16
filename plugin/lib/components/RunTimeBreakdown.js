'use babel';

import React from 'react';

import RunTimePerfBar from './RunTimePerfBar';
import PerfBarContainer from './generic/PerfBarContainer';
import OperationInfoStore from '../stores/operationinfo_store';
import PerfVisState from '../models/PerfVisState';

const COLOR_CLASSES = [
  'innpv-blue-color-1',
  'innpv-blue-color-2',
  'innpv-blue-color-3',
  'innpv-blue-color-4',
  'innpv-blue-color-5',
];

export default class RunTimeBreakdown extends React.Component {
  constructor(props) {
    super(props);

    const operationInfos = OperationInfoStore.getOperationInfos();
    this.state = {
      operationInfos,
      totalTimeUs: this._getTotalTimeUs(operationInfos),
      marginTop: 0,
    };

    this._onStoreChange = this._onStoreChange.bind(this);
    this._perfBarGenerator = this._perfBarGenerator.bind(this);
    this._updateMarginTop = this._updateMarginTop.bind(this);
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

  _updateMarginTop(marginTop) {
    this.setState({marginTop});
  }

  render() {
    const {perfVisState} = this.props;
    const {operationInfos, marginTop} = this.state;
    const disabled = perfVisState === PerfVisState.DEBOUNCING ||
      (perfVisState === PerfVisState.ANALYZING && operationInfos.length == 0);

    return (
      <PerfBarContainer disabled={disabled} marginTop={marginTop}>
        {operationInfos.map((opInfo, index) => (
          <RunTimePerfBar
            key={opInfo.getBoundName()}
            operationInfo={opInfo}
            percentage={opInfo.getRuntimeUs() / this.state.totalTimeUs * 100}
            colorClass={COLOR_CLASSES[index % COLOR_CLASSES.length]}
            updateMarginTop={this._updateMarginTop}
          />
        ))}
      </PerfBarContainer>
    );
  }
}
