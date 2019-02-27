'use babel';

import React from 'react';

export default class PerfBar extends React.Component {
  render() {
    return (
      <div
        className={`innpv-perfbar ${this.props.colorClass}`}
        style={{height: `${this.props.percentage}%`}}
      />
    );
  }
}
