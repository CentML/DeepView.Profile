'use babel';

import React from 'react';

export default class NumericDisplay extends React.Component {
  render() {
    const {top, number, bottom} = this.props;
    return (
      <div className="innpv-numericdisplay">
        <div className="innpv-numericdisplay-top">{top}</div>
        <div className="innpv-numericdisplay-number">{number}</div>
        <div className="innpv-numericdisplay-bottom">{bottom}</div>
      </div>
    );
  }
}
