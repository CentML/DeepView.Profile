'use babel';

import React from 'react';

import PerfBarContainer from './PerfBarContainer';
import PerfVisStatusBar from './PerfVisStatusBar';
import PerfVisState from '../models/PerfVisState';

function PerfVisHeader() {
  return (
    <div className="innpv-header">
      <span className="icon icon-graph"></span>innpv
    </div>
  );
}

export default class PerfVisMainView extends React.Component {
  _classes() {
    switch (this.props.perfVisState) {
      case PerfVisState.ANALYZING:
      case PerfVisState.DEBOUNCING:
        return "innpv-contents innpv-no-events";

      default:
        return "innpv-contents";
    }
  }

  render() {
    return (
      <div className="innpv-main">
        <PerfVisHeader />
        <div className={this._classes()}>
          <PerfBarContainer editor={this.props.editor} />
        </div>
        <PerfVisStatusBar perfVisState={this.props.perfVisState} />
      </div>
    );
  }
}
