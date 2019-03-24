'use babel';

import React from 'react';

import PerfBar from './PerfBar';
import OperationInfoStore from '../stores/operationinfo_store';
import PerfVisState from '../models/PerfVisState';

const COLOR_CLASSES = [
  'innpv-bar-color-1',
  'innpv-bar-color-2',
  'innpv-bar-color-3',
  'innpv-bar-color-4',
  'innpv-bar-color-5',
];

export default class PerfBarContainer extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      operationInfos: OperationInfoStore.getOperationInfos(),
      marginTop: 0,
    };

    this._onStoreChange = this._onStoreChange.bind(this);
    this._updateMarginTop = this._updateMarginTop.bind(this);
  }

  componentDidMount() {
    OperationInfoStore.addListener(this._onStoreChange);
  }

  componentWillUnmount() {
    OperationInfoStore.removeListener(this._onStoreChange);
  }

  _onStoreChange() {
    this.setState({
      operationInfos: OperationInfoStore.getOperationInfos(),
    });
  }

  _updateMarginTop(marginTop) {
    this.setState({marginTop});
  }

  _classes() {
    const {perfVisState} = this.props;
    const mainClass = 'innpv-perfbarcontainer-wrap';

    if (perfVisState === PerfVisState.DEBOUNCING ||
        (perfVisState === PerfVisState.ANALYZING &&
          this.state.operationInfos.length == 0)) {
      return mainClass + ' innpv-no-events';
    }
    return mainClass;
  }

  render() {
    const totalTimeMicro =
      this.state.operationInfos.reduce((acc, info) => acc + info.getRuntimeUs(), 0);

    return (
      <div className={this._classes()}>
        <div className="innpv-perfbarcontainer">
          <div
            className="innpv-perfbarcontainer-inner"
            style={{marginTop: `-${this.state.marginTop}px`}}
          >
            {this.state.operationInfos.map(
              (operationInfo, index) =>
                <PerfBar
                  key={operationInfo.getBoundName()}
                  operationInfo={operationInfo}
                  percentage={operationInfo.getRuntimeUs() / totalTimeMicro * 100}
                  colorClass={COLOR_CLASSES[index % COLOR_CLASSES.length]}
                  updateMarginTop={this._updateMarginTop}
                />
            )}
          </div>
        </div>
      </div>
    );
  }
}
