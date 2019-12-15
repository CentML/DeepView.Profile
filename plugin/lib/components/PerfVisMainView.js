'use babel';

import React from 'react';

import ErrorMessage from './ErrorMessage';
import Memory from './Memory';
import RunTimeBreakdown from './RunTimeBreakdown';
import PerfVisStatusBar from './PerfVisStatusBar';
import Throughput from './Throughput';
import PerfVisState from '../models/PerfVisState';
import BatchSizeStore from '../stores/batchsize_store';
import AnalysisStore from '../stores/analysis_store';
import INNPVStore from '../stores/innpv_store';
import SourceMarker from '../marker';

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
      inputInfo: BatchSizeStore.getInputInfo(),
      overallMemoryUsage: AnalysisStore.getOverallMemoryUsage(),
    };
    this._onStoreUpdate = this._onStoreUpdate.bind(this);
    this._handleStatusBarClick = this._handleStatusBarClick.bind(this);
    this._handleSliderHoverEnter = this._handleSliderHoverEnter.bind(this);
    this._handleSliderHoverExit = this._handleSliderHoverExit.bind(this);
  }

  componentDidMount() {
    BatchSizeStore.addListener(this._onStoreUpdate);
    AnalysisStore.addListener(this._onStoreUpdate);
  }

  componentWillUnmount() {
    BatchSizeStore.removeListener(this._onStoreUpdate);
    AnalysisStore.removeListener(this._onStoreUpdate);
  }

  _onStoreUpdate() {
    this.setState({
      throughput: BatchSizeStore.getThroughputModel(),
      inputInfo: BatchSizeStore.getInputInfo(),
      overallMemoryUsage: AnalysisStore.getOverallMemoryUsage(),
    });
  }

  _handleStatusBarClick() {
    if (this.props.perfVisState !== PerfVisState.SHOWING_PREDICTIONS) {
      return;
    }
    BatchSizeStore.clearPredictions();
    INNPVStore.setPerfVisState(PerfVisState.READY);
  }

  _handleSliderHoverEnter() {
    // TODO: Use these event handling functions to highlight the batch size
  }

  _handleSliderHoverExit() {
  }

  _subrowClasses() {
    const {perfVisState} = this.props;
    const {throughput, memory} = this.state;
    const mainClass = 'innpv-contents-subrows';
    if (perfVisState === PerfVisState.DEBOUNCING ||
        (perfVisState === PerfVisState.ANALYZING &&
          (throughput == null || memory == null))) {
      return mainClass + ' innpv-no-events';
    }
    return mainClass;
  }

  _renderBody() {
    const {perfVisState} = this.props;
    if (this.props.errorMessage !== '') {
      return <ErrorMessage perfVisState={perfVisState} message={this.props.errorMessage} />;
    } else {
      return (
        <div className="innpv-contents-columns">
          <div className="innpv-perfbar-contents">
            <RunTimeBreakdown perfVisState={perfVisState} />
          </div>
          <div className={this._subrowClasses()}>
            <Throughput
              model={this.state.throughput}
              handleSliderHoverEnter={this._handleSliderHoverEnter}
              handleSliderHoverExit={this._handleSliderHoverExit}
            />
            <Memory
              model={this.state.overallMemoryUsage}
              handleSliderHoverEnter={this._handleSliderHoverEnter}
              handleSliderHoverExit={this._handleSliderHoverExit}
            />
          </div>
        </div>
      );
    }
  }

  render() {
    return (
      <div className="innpv-main">
        <PerfVisHeader />
        <div className="innpv-contents">{this._renderBody()}</div>
        <PerfVisStatusBar
          handleClick={this._handleStatusBarClick}
          perfVisState={this.props.perfVisState}
        />
      </div>
    );
  }
}
