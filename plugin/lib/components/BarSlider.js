'use babel';

import React from 'react';

class BarSlider extends React.Component {
  render() {
    const {percentage, limit} = this.props;
    return (
      <div className="innpv-barslider">
        <div className="innpv-barslider-barwrap">
          <div
            className="innpv-barslider-bar"
            style={{height: `${percentage}%`}}
          />
          {limit > 0 ? <div className="innpv-barslider-limit" style={{height: `${limit}%`}} /> : null}
        </div>
      </div>
    );
  }
}

BarSlider.defaultProps = {
  limit: 0,
};

export default BarSlider;
