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
      annotationLocation: BatchSizeStore.getAnnotationLocation(),
    };
    this._annotation_marker = new SourceMarker(this.props.editor);
    this._onStoreUpdate = this._onStoreUpdate.bind(this);
    this._handleStatusBarClick = this._handleStatusBarClick.bind(this);
    this._handleSliderHoverEnter = this._handleSliderHoverEnter.bind(this);
    this._handleSliderHoverExit = this._handleSliderHoverExit.bind(this);
  }

  componentDidMount() {
    BatchSizeStore.addListener(this._onStoreUpdate);
    this._annotation_marker.register(this.state.annotationLocation);
  }

  componentDidUpdate(prevProps, prevState) {
    this._annotation_marker.reconcileLocation(
      prevState.annotationLocation,
      this.state.annotationLocation,
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
      annotationLocation: BatchSizeStore.getAnnotationLocation(),
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
        <div className={this._classes()}>{this._renderBody()}</div>
        <PerfVisStatusBar
          handleClick={this._handleStatusBarClick}
          perfVisState={this.props.perfVisState}
        />
      </div>
    );
  }
}
