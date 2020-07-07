'use babel';

import React from 'react';
import {connect} from 'react-redux';

import AppState from '../models/AppState';
import GetStarted from './GetStarted';
import PerfVisMainView from './PerfVisMainView';

class PerfVis extends React.Component {
  _renderContents() {
    const {
      appState,
      perfVisState,
      errorMessage,
      errorFilePath,
      errorLineNumber,
      projectRoot,
      handleGetStartedClick,
      triggerProfiling,
      initialHost,
      initialPort,
    } = this.props;

    switch (appState) {
      case AppState.OPENED:
      case AppState.CONNECTING:
        return (
          <GetStarted
            appState={appState}
            handleClick={handleGetStartedClick}
            errorMessage={errorMessage}
            initialHost={initialHost}
            initialPort={initialPort}
          />
        );

      case AppState.CONNECTED:
        return (
          <PerfVisMainView
            perfVisState={perfVisState}
            errorMessage={errorMessage}
            errorFilePath={errorFilePath}
            errorLineNumber={errorLineNumber}
            projectRoot={projectRoot}
            triggerProfiling={triggerProfiling}
          />
        );

      default:
        return null;
    }
  }

  render() {
    return <div className="innpv-wrap">{this._renderContents()}</div>;
  }
}

const mapStateToProps = (state, ownProps) => ({
  appState: state.appState,
  perfVisState: state.perfVisState,
  errorMessage: state.errorMessage,
  errorFilePath: state.errorFilePath,
  errorLineNumber: state.errorLineNumber,
  projectRoot: state.projectRoot,
  ...ownProps,
});

export default connect(mapStateToProps)(PerfVis);
