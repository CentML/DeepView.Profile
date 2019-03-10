'use babel';

import React from 'react';

function isNumeric(candidate) {
  return !isNaN(parseFloat(candidate));
}

class NumericDisplay extends React.Component {
  render() {
    const {top, number, bottom, precision} = this.props;
    return (
      <div className="innpv-numericdisplay">
        <div className="innpv-numericdisplay-top">{top}</div>
        <div className="innpv-numericdisplay-number">
          {isNumeric(number) ? number.toFixed(precision) : number}
        </div>
        <div className="innpv-numericdisplay-bottom">{bottom}</div>
      </div>
    );
  }
}

NumericDisplay.defaultProps = {
  precision: 0,
};

export default NumericDisplay;
