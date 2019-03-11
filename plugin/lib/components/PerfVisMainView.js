'use babel';

import React from 'react';

import ErrorMessage from './ErrorMessage';
import Memory from './Memory';
import PerfBarContainer from './PerfBarContainer';
import PerfVisStatusBar from './PerfVisStatusBar';
import Throughput from './Throughput';
import PerfVisState from '../models/PerfVisState';
import BatchSizeStore from '../stores/batchsize_store';
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
      throughput: BatchSizeStore.getThroughputModel(),
      memory: BatchSizeStore.getMemoryModel(),
    };
    this._onStoreUpdate = this._onStoreUpdate.bind(this);
    this._handleStatusBarClick = this._handleStatusBarClick.bind(this);
  }

  componentDidMount() {
    BatchSizeStore.addListener(this._onStoreUpdate);
  }

  componentWillUnmount() {
    BatchSizeStore.removeListener(this._onStoreUpdate);
  }

  _onStoreUpdate() {
    this.setState({
      throughput: BatchSizeStore.getThroughputModel(),
      memory: BatchSizeStore.getMemoryModel(),
    });
  }

  _handleStatusBarClick() {
    if (this.props.perfVisState !== PerfVisState.SHOWING_PREDICTIONS) {
      return;
    }
    BatchSizeStore.clearPredictions();
    INNPVStore.setPerfVisState(PerfVisState.READY);
  }

  _classes() {
    switch (this.props.perfVisState) {
      case PerfVisState.ANALYZING:
      case PerfVisState.DEBOUNCING:
        return "innpv-contents innpv-no-events";

      default:
        return "innpv-contents";
    }
  }

  _renderBody() {
    if (this.props.errorMessage !== '') {
      return <ErrorMessage message={this.props.errorMessage} />;
    } else {
      return (
        <div className="innpv-contents-columns">
          <PerfBarContainer editor={this.props.editor} />
          <div className="innpv-contents-subrows">
            <Throughput model={this.state.throughput} />
            <Memory model={this.state.memory} />
          </div>
        </div>
      );
    }
  }

  render() {
    return (
      <div className="innpv-main">
        <PerfVisHeader />
        <div className={this._classes()}>{this._renderBody()}</div>
        <PerfVisStatusBar
          handleClick={this._handleStatusBarClick}
          perfVisState={this.props.perfVisState}
        />
      </div>
    );
  }
}
