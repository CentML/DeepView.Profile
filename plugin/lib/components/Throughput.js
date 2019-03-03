'use babel';

import React from 'react';

import Subheader from './Subheader';

export default class Throughput extends React.Component {
  render() {
    return (
      <div className="innpv-throughput innpv-subpanel">
        <Subheader icon="flame">Training Throughput</Subheader>
      </div>
    );
  }
}
