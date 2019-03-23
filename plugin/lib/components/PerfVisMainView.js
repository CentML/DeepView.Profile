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
      memory: BatchSizeStore.getMemoryModel(),
      inputInfo: BatchSizeStore.getInputInfo(),
    };
    this._annotation_marker = new SourceMarker(INNPVStore.getEditor());
    this._onStoreUpdate = this._onStoreUpdate.bind(this);
    this._handleStatusBarClick = this._handleStatusBarClick.bind(this);
    this._handleSliderHoverEnter = this._handleSliderHoverEnter.bind(this);
    this._handleSliderHoverExit = this._handleSliderHoverExit.bind(this);
  }

  componentDidMount() {
    const {inputInfo} = this.state;
    BatchSizeStore.addListener(this._onStoreUpdate);
    this._annotation_marker.register(inputInfo && inputInfo.getAnnotationStart());
  }

  componentDidUpdate(prevProps, prevState) {
    const {inputInfo} = this.state;
    this._annotation_marker.reconcileLocation(
      prevState.inputInfo && prevState.inputInfo.getAnnotationStart(),
      inputInfo && inputInfo.getAnnotationStart(),
    );
  }

  componentWillUnmount() {
    BatchSizeStore.removeListener(this._onStoreUpdate);
    this._annotation_marker.remove();
  }

  _onStoreUpdate() {
    this.setState({
      throughput: BatchSizeStore.getThroughputModel(),
      memory: BatchSizeStore.getMemoryModel(),
      inputInfo: BatchSizeStore.getInputInfo(),
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
    this._annotation_marker.showDecoration({type: 'line', class: 'innpv-line-highlight'});
  }

  _handleSliderHoverExit() {
    this._annotation_marker.hideDecoration();
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
    if (this.props.errorMessage !== '') {
      return <ErrorMessage message={this.props.errorMessage} />;
    } else {
      const {perfVisState} = this.props;
      return (
        <div className="innpv-contents-columns">
          <PerfBarContainer perfVisState={perfVisState} />
          <div className={this._subrowClasses()}>
            <Throughput
              model={this.state.throughput}
              handleSliderHoverEnter={this._handleSliderHoverEnter}
              handleSliderHoverExit={this._handleSliderHoverExit}
            />
            <Memory
              model={this.state.memory}
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
