'use babel';

import React from 'react';

import GetStarted from './GetStarted';
import AppState from '../models/AppState';
import INNPVStore from '../stores/innpv_store';

export default class PerfVis extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      appState: INNPVStore.getAppState(),
    };
    this._onStoreUpdate = this._onStoreUpdate.bind(this);
  }

  componentDidMount() {
    INNPVStore.addListener(this._onStoreUpdate);
  }

  componentWillUnmount() {
    INNPVStore.removeListener(this._onStoreUpdate);
  }

  _onStoreUpdate() {
    this.setState({appState: INNPVStore.getAppState()});
  }

  render() {
    return (
      <div className="innpv-main">
        {this.state.appState === AppState.OPENED ?
            <GetStarted handleClick={this.props.handleGetStartedClick} /> : null}
      </div>
    );
  }
}
