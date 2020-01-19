'use babel';

import React from 'react';

import ErrorMessage from './ErrorMessage';
import Memory from './Memory';
import MemoryBreakdown from './MemoryBreakdown';
import PerfVisStatusBar from './PerfVisStatusBar';
import RunTimeBreakdown from './RunTimeBreakdown';
import Throughput from './Throughput';
import PerfVisState from '../models/PerfVisState';
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
    this._handleStatusBarClick = this._handleStatusBarClick.bind(this);
    this._handleSliderHoverEnter = this._handleSliderHoverEnter.bind(this);
    this._handleSliderHoverExit = this._handleSliderHoverExit.bind(this);
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
    const mainClass = 'innpv-contents-subrows';
    if (perfVisState === PerfVisState.MODIFIED ||
        perfVisState === PerfVisState.ANALYZING) {
      return mainClass + ' innpv-no-events';
    }
    return mainClass;
  }

  _renderBody() {
    const {perfVisState, projectRoot, errorMessage} = this.props;
    if (this.props.errorMessage !== '') {
      return <ErrorMessage perfVisState={perfVisState} message={errorMessage} />;
    } else {
      return (
        <div className="innpv-contents-columns">
          <div className="innpv-perfbar-contents">
            <RunTimeBreakdown perfVisState={perfVisState} projectRoot={projectRoot} />
            <MemoryBreakdown perfVisState={perfVisState} projectRoot={projectRoot} />
          </div>
          <div className={this._subrowClasses()}>
            <Throughput
              handleSliderHoverEnter={this._handleSliderHoverEnter}
              handleSliderHoverExit={this._handleSliderHoverExit}
            />
            <Memory
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
