'use babel';

import React from 'react';

import PerfBarContainer from './PerfBarContainer';
import PerfVisStatusBar from './PerfVisStatusBar';
import PerfVisState from '../models/PerfVisState';
import INNPVStore from '../stores/innpv_store';

function PerfVisHeader() {
  return (
    <div className="innpv-header">
      <span className="icon icon-graph"></span>innpv
    </div>
  );
}

export default class PerfVisMainView extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      perfVisState: INNPVStore.getPerfVisState(),
    };

    this._onStoreUpdate = this._onStoreUpdate.bind(this);
  }

  componentDidMount() {
    INNPVStore.addListener(this._onStoreUpdate);
  }

  componentWillUnmount() {
    INNPVStore.removeListener(this._onStoreUpdate);
  }

  _onStoreUpdate() {
    this.setState({perfVisState: INNPVStore.getPerfVisState()});
  }

  _classes() {
    switch (this.state.perfVisState) {
      case PerfVisState.ANALYZING:
      case PerfVisState.DEBOUNCING:
        return "innpv-contents innpv-no-events";

      default:
        return "innpv-contents";
    }
  }

  render() {
    return (
      <div className="innpv-main">
        <PerfVisHeader />
        <div className={this._classes()}>
          <PerfBarContainer
            operationInfos={this.props.operationInfos}
            editor={this.props.editor}
          />
        </div>
        <PerfVisStatusBar perfVisState={this.state.perfVisState} />
      </div>
    );
  }
}
