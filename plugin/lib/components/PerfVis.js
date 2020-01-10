'use babel';

import React from 'react';
import {connect} from 'react-redux';

import AppState from '../models/AppState';
import GetStarted from './GetStarted';
import PerfVisMainView from './PerfVisMainView';

class PerfVis extends React.Component {
  _renderContents() {
    const {appState, perfVisState, errorMessage, handleGetStartedClick} = this.props;

    switch (appState) {
      case AppState.OPENED:
      case AppState.CONNECTING:
        return (
          <GetStarted
            appState={appState}
            handleClick={handleGetStartedClick}
            errorMessage={errorMessage}
          />
        );

      case AppState.CONNECTED:
        return (
          <PerfVisMainView
            perfVisState={perfVisState}
            errorMessage={errorMessage}
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

const mapStateToProps = (state) => ({
  appState: state.appState,
  perfVisState: state.perfVisState,
  errorMessage: state.errorMessage,
});

export default connect(mapStateToProps)(PerfVis);
