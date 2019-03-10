'use babel';

import React from 'react';

import Subheader from './Subheader';
import BarSlider from './BarSlider';
import NumericDisplay from './NumericDisplay';

export default class Throughput extends React.Component {
  render() {
    const {model} = this.props;
    const notReady = model == null;

    return (
      <div className="innpv-throughput innpv-subpanel">
        <Subheader icon="flame">Training Throughput</Subheader>
        <div className="innpv-subpanel-content">
          <BarSlider
            percentage={notReady ? 0 : model.displayPct}
            limitPercentage={notReady ? 100 : model.limitPct}
          />
          <div className="innpv-subpanel-sidecontent">
            <NumericDisplay
              top="Throughput"
              number={notReady ? '---' : model.throughput}
              bottom="samples/second"
            />
            <div className="innpv-separator" />
            <NumericDisplay
              top="Theoretical Maximum"
              number={notReady ? '---' : model.maxThroughput}
              bottom="samples/second"
            />
          </div>
        </div>
      </div>
    );
  }
}
