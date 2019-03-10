'use babel';

import React from 'react';

import Resizable from './resizable';

class BarSlider extends React.Component {
  constructor(props) {
    super(props);
    this._handleResize = this._handleResize.bind(this);
  }

  _handleResize(diff) {
  }

  render() {
    const {percentage, limitPercentage} = this.props;
    const limitBarHeight = 100 - limitPercentage;
    return (
      <div className="innpv-barslider">
        <div className="innpv-barslider-barwrap">
          <Resizable
            className="innpv-resizable-barslider"
            heightPct={percentage}
            handleResize={this._handleResize}
          >
            <div className="innpv-barslider-bar" />
          </Resizable>
          {limitBarHeight > 1e-3 ?
            <div className="innpv-barslider-limit" style={{height: `${limitBarHeight}%`}} /> : null}
        </div>
      </div>
    );
  }
}

BarSlider.defaultProps = {
  limitPercentage: 100,
};

export default BarSlider;
