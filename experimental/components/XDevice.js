'use babel';

import React from 'react';

import Subheader from './Subheader';

function XDeviceBar(props) {
  return (
    <div
      className={`innpv-xdevice-bar innpv-xdevice-bar-color-${props.color}`}
      style={{width: `${props.width}px`}}
    >
      <span className="innpv-xdevice-device">{props.device}</span>
      <span>{props.speedup}</span>
    </div>
  );
}

export default class XDevice extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div className="innpv-xdevice innpv-subpanel">
        <Subheader icon="dashboard">Throughput Predictions</Subheader>
        <div className="innpv-subpanel-content">
          <div className="innpv-xdevice-bars">
            <XDeviceBar device="P4" speedup="0.5x" width={55} color={5} />
            <XDeviceBar device="T4" speedup="0.7x" width={77} color={4} />
            <XDeviceBar device="2070 (*)" speedup="1.0x" width={110} color={3} />
            <XDeviceBar device="P100" speedup="1.3x" width={143} color={2} />
            <XDeviceBar device="V100" speedup="2.0x" width={220} color={1} />
          </div>
        </div>
      </div>
    );
  }
}

