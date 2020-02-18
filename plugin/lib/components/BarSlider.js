'use babel';

import React from 'react';

import Resizable from './Resizable';

class BarSlider extends React.Component {
  render() {
    const {
      percentage,
      limitPercentage,
      onMouseEnter,
      onMouseLeave,
      onClick,
    } = this.props;
    const limitBarHeight = 100 - limitPercentage;
    return (
      <div
        className="innpv-barslider"
        onClick={onClick}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
      >
        <div className="innpv-barslider-barwrap">
          <Resizable
            className="innpv-resizable-barslider"
            heightPct={percentage}
            handleResize={this.props.handleResize}
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
  handleResize: () => {},
  onClick: () => {},
  onMouseEnter: () => {},
  onMouseLeave: () => {},
};

export default BarSlider;
