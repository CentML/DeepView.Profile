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
    const {percentage, limit} = this.props;
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
