'use babel';

import React from 'react';

import PerfVisState from '../models/PerfVisState';

function LoadingIcon() {
  return (
    <div className="innpv-statusbar-loading">
      <span className="loading loading-spinner-tiny inline-block" />
    </div>
  );
}

function ErrorIcon() {
  return (
    <div className="innpv-statusbar-icon">
      <span className="icon icon-alert" />
    </div>
  );
}

function SyncButton(props) {
  return (
    <div onClick={props.handleClick} className="innpv-statusbar-icon innpv-clickable">
      <span className="icon icon-x" />
    </div>
  );
}

class PerfVisStatusBar extends React.Component {
  _getMessage() {
    switch (this.props.perfVisState) {
      case PerfVisState.READY:
        return 'Ready';

      case PerfVisState.ERROR:
        return 'Analysis error';

      case PerfVisState.ANALYZING:
        return 'Analyzing...';

      case PerfVisState.SHOWING_PREDICTIONS:
        return 'Showing predicted performance';

      case PerfVisState.MODIFIED:
        return 'Unsaved changes; save to re-enable interactivity';

      case PerfVisState.EXPLORING_WEIGHTS:
      case PerfVisState.EXPLORING_OPERATIONS:
        return 'Showing breakdown details';
    }
  }

  _renderIcon() {
    switch (this.props.perfVisState) {
      case PerfVisState.ERROR:
        return <ErrorIcon />;

      case PerfVisState.ANALYZING:
        return <LoadingIcon />;

      case PerfVisState.SHOWING_PREDICTIONS:
        return <SyncButton handleClick={this.props.handleClick} />;

      default:
        return null;
    }
  }

  render() {
    return (
      <div className="innpv-statusbar">
        <div className="innpv-statusbar-message">{this._getMessage()}</div>
        {this._renderIcon()}
      </div>
    );
  }
}

PerfVisStatusBar.defaultProps = {
  handleClick: () => {},
};

export default PerfVisStatusBar;
