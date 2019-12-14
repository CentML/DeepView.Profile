'use babel';

import React from 'react';

function isNumeric(candidate) {
  return !isNaN(parseFloat(candidate));
}

class NumericDisplay extends React.Component {
  _numberToDisplay() {
    const {number, precision} = this.props;
    if (!isNumeric(number)) {
      return number;
    }

    const fixed = number.toFixed(precision);
    const zero = (0).toFixed(precision);
    if (number > 0 && fixed === zero) {
      return '< 1';
    }

    return fixed;
  }

  render() {
    const {top, bottom} = this.props;
    return (
      <div className="innpv-numericdisplay">
        <div className="innpv-numericdisplay-top">{top}</div>
        <div className="innpv-numericdisplay-number">
          {this._numberToDisplay()}
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
