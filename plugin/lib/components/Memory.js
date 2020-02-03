'use babel';

import React from 'react';
import {connect} from 'react-redux';

import Subheader from './Subheader';
import BarSlider from './BarSlider';
import NumericDisplay from './NumericDisplay';
import PerfVisState from '../models/PerfVisState';
import {toPercentage} from '../utils';

export default class Memory extends React.Component {
  constructor(props) {
    super(props);
    this._handleResize = this._handleResize.bind(this);
  }

  _handleResize(deltaPct, basePct) {
    // TODO: Add in memory predictions again
  }

  _toMb(bytes) {
    return bytes / 1024.0 / 1024.0;
  }

  render() {
    const {breakdown, handleSliderHoverEnter, handleSliderHoverExit} = this.props;
    const notReady = breakdown == null;
    const percentage = notReady ? 0 : toPercentage(
      breakdown.peakUsageBytes, breakdown.memoryCapacityBytes);

    return (
      <div className="innpv-memory innpv-subpanel">
        <Subheader icon="database">Peak Memory Usage</Subheader>
        <div className="innpv-subpanel-content">
          <BarSlider
            percentage={percentage}
            handleResize={this._handleResize}
            onMouseEnter={handleSliderHoverEnter}
            onMouseLeave={handleSliderHoverExit}
          />
          <div className="innpv-subpanel-sidecontent">
            <NumericDisplay
              top="Peak Usage"
              number={notReady ? '---' : this._toMb(breakdown.peakUsageBytes)}
              bottom="Megabytes"
            />
            <div className="innpv-separator" />
            <NumericDisplay
              top="Maximum Capacity"
              number={notReady ? '---' : this._toMb(breakdown.memoryCapacityBytes)}
              bottom="Megabytes"
            />
          </div>
        </div>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => ({
  breakdown: state.breakdown.model,
  ...ownProps,
});

export default connect(mapStateToProps)(Memory);
