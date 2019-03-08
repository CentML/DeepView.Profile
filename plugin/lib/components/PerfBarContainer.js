'use babel';

import React from 'react';

import PerfBar from './PerfBar';
import OperationInfoStore from '../stores/operationinfo_store';

const COLOR_CLASSES = [
  'ui-site-1',
  'ui-site-2',
  'ui-site-3',
  'ui-site-4',
  'ui-site-5',
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

  render() {
    const totalTimeMicro =
      this.state.operationInfos.reduce((acc, info) => acc + info.getRuntimeUs(), 0);

    return (
      <div className="innpv-perfbarcontainer-wrap">
        <div className="innpv-perfbarcontainer">
          <div
            className="innpv-perfbarcontainer-inner"
            style={{marginTop: `-${this.state.marginTop}px`}}
          >
            {this.state.operationInfos.map(
              (operationInfo, index) =>
                <PerfBar
                  key={operationInfo.getBoundName()}
                  editor={this.props.editor}
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
