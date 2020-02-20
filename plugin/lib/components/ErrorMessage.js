'use babel';

import React from 'react';

import PerfVisState from '../models/PerfVisState';

export default class ErrorMessage extends React.Component {
  _classes() {
    const mainClass = 'innpv-error';
    const {perfVisState, projectModified} = this.props;
    if (projectModified || perfVisState === PerfVisState.ANALYZING) {
      return mainClass + ' innpv-no-events';
    }
    return mainClass;
  }

  render() {
    return (
      <div className={this._classes()}>
        <div className="innpv-error-inner">
          <h1>Analysis Error</h1>
          <p>{this.props.message}</p>
        </div>
      </div>
    );
  }
}
