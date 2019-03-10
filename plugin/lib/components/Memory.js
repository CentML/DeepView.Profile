'use babel';

import React from 'react';

import Subheader from './Subheader';
import BarSlider from './BarSlider';
import NumericDisplay from './NumericDisplay';

export default class Memory extends React.Component {
  render() {
    const {model} = this.props;
    const notReady = model == null;

    return (
      <div className="innpv-memory innpv-subpanel">
        <Subheader icon="database">Peak Memory Usage</Subheader>
        <div className="innpv-subpanel-content">
          <BarSlider percentage={notReady ? 0 : model.displayPct} />
          <div className="innpv-subpanel-sidecontent">
            <NumericDisplay
              top="Peak Usage"
              number={notReady ? '---' : model.usage}
              bottom="Megabytes"
            />
            <div className="innpv-separator" />
            <NumericDisplay
              top="Maximum Capacity"
              number={notReady ? '---' : model.maxCapacity}
              bottom="Megabytes"
            />
          </div>
        </div>
      </div>
    );
  }
}

