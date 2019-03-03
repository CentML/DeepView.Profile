'use babel';

import React from 'react';

import Subheader from './Subheader';
import BarSlider from './BarSlider';
import NumericDisplay from './NumericDisplay';

export default class Throughput extends React.Component {
  render() {
    return (
      <div className="innpv-throughput innpv-subpanel">
        <Subheader icon="flame">Training Throughput</Subheader>
        <div className="innpv-subpanel-content">
          <BarSlider percentage={70} limit={10} />
          <div className="innpv-subpanel-sidecontent">
            <NumericDisplay
              top="Throughput"
              number={1313}
              bottom="samples/second"
            />
            <div className="innpv-separator" />
            <NumericDisplay
              top="Theoretical Maximum"
              number={2015}
              bottom="samples/second"
            />
          </div>
        </div>
      </div>
    );
  }
}
