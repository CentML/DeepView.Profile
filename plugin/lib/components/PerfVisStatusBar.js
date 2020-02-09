'use babel';

import React from 'react';
import {connect} from 'react-redux';

import PerfVisState from '../models/PerfVisState';
import AnalysisActions from '../redux/actions/analysis';

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
    <div className="innpv-statusbar-icon innpv-clickable">
      <span className="icon icon-x" />
    </div>
  );
}

function ExplorationButton(props) {
  const {Fragment} = React;
  return (
    <div className="innpv-statusbar-iconbar">
      <div
        onClick={props.explorePrevious}
        className="innpv-statusbar-icon innpv-clickable"
      >
        <span className="icon icon-chevron-left" />
      </div>
      <div
        onClick={props.clearExplored}
        className="innpv-statusbar-icon innpv-clickable"
      >
        <span className="icon icon-x" />
      </div>
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
        return 'Showing weight breakdown details';

      case PerfVisState.EXPLORING_OPERATIONS:
        return 'Showing operation breakdown details';
    }
  }

  _renderIcon() {
    switch (this.props.perfVisState) {
      case PerfVisState.ERROR:
        return <ErrorIcon />;

      case PerfVisState.ANALYZING:
        return <LoadingIcon />;

      case PerfVisState.SHOWING_PREDICTIONS:
        return <SyncButton />;

      case PerfVisState.EXPLORING_WEIGHTS:
      case PerfVisState.EXPLORING_OPERATIONS:
        return (
          <ExplorationButton
            explorePrevious={this.props.explorePrevious}
            clearExplored={this.props.clearExplored}
          />
        );

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

const mapDispatchToProps = (dispatch) => ({
  explorePrevious: () => dispatch(AnalysisActions.explorePrevious()),
  clearExplored: () => dispatch(AnalysisActions.clearExplored()),
});

export default connect(null, mapDispatchToProps)(PerfVisStatusBar);
