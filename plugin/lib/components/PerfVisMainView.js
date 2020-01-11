'use babel';

import React from 'react';

import ErrorMessage from './ErrorMessage';
import Memory from './Memory';
import MemoryBreakdown from './MemoryBreakdown';
import PerfVisStatusBar from './PerfVisStatusBar';
import Throughput from './Throughput';
import PerfVisState from '../models/PerfVisState';
import AnalysisStore from '../stores/analysis_store';
import INNPVStore from '../stores/innpv_store';
import SourceMarker from '../editor/marker';

function PerfVisHeader() {
  return (
    <div className="innpv-header">
      <span className="icon icon-graph"></span>Skyline
    </div>
  );
}

export default class PerfVisMainView extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      overallMemoryUsage: AnalysisStore.getOverallMemoryUsage(),
      throughput: AnalysisStore.getThroughput(),
    };
    this._onStoreUpdate = this._onStoreUpdate.bind(this);
    this._handleStatusBarClick = this._handleStatusBarClick.bind(this);
    this._handleSliderHoverEnter = this._handleSliderHoverEnter.bind(this);
    this._handleSliderHoverExit = this._handleSliderHoverExit.bind(this);
  }

  componentDidMount() {
    AnalysisStore.addListener(this._onStoreUpdate);
  }

  componentWillUnmount() {
    AnalysisStore.removeListener(this._onStoreUpdate);
  }

  _onStoreUpdate() {
    this.setState({
      overallMemoryUsage: AnalysisStore.getOverallMemoryUsage(),
      throughput: AnalysisStore.getThroughput(),
    });
  }

  _handleStatusBarClick() {
    // TODO: Handle status bar clicks (currently only to undo predictions)
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
    if (perfVisState === PerfVisState.MODIFIED ||
        (perfVisState === PerfVisState.ANALYZING &&
          (throughput == null || memory == null))) {
      return mainClass + ' innpv-no-events';
    }
    return mainClass;
  }

  _renderBody() {
    const {perfVisState, projectRoot} = this.props;
    if (this.props.errorMessage !== '') {
      return <ErrorMessage perfVisState={perfVisState} message={this.props.errorMessage} />;
    } else {
      return (
        <div className="innpv-contents-columns">
          <div className="innpv-perfbar-contents">
            <MemoryBreakdown perfVisState={perfVisState} projectRoot={projectRoot} />
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
