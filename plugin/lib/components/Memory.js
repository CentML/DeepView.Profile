'use babel';

import React from 'react';

import Subheader from './Subheader';
import BarSlider from './BarSlider';
import NumericDisplay from './NumericDisplay';
import BatchSizeStore from '../stores/batchsize_store';
import INNPVStore from '../stores/innpv_store';
import PerfVisState from '../models/PerfVisState';

export default class Memory extends React.Component {
  constructor(props) {
    super(props);
    this._handleResize = this._handleResize.bind(this);
  }

  _handleResize(deltaPct, basePct) {
    // TODO: Add in memory predictions again
    return;

    BatchSizeStore.updateMemoryUsage(deltaPct, basePct);
    INNPVStore.setPerfVisState(PerfVisState.SHOWING_PREDICTIONS);
  }

  render() {
    const {model, handleSliderHoverEnter, handleSliderHoverExit} = this.props;
    const notReady = model == null;

    return (
      <div className="innpv-memory innpv-subpanel">
        <Subheader icon="database">Peak Memory Usage</Subheader>
        <div className="innpv-subpanel-content">
          <BarSlider
            percentage={notReady ? 0 : model.displayPct}
            handleResize={this._handleResize}
            onMouseEnter={handleSliderHoverEnter}
            onMouseLeave={handleSliderHoverExit}
          />
          <div className="innpv-subpanel-sidecontent">
            <NumericDisplay
              top="Peak Usage"
              number={notReady ? '---' : model.peakUsageMb}
              bottom="Megabytes"
            />
            <div className="innpv-separator" />
            <NumericDisplay
              top="Maximum Capacity"
              number={notReady ? '---' : model.memoryCapacityMb}
              bottom="Megabytes"
            />
          </div>
        </div>
      </div>
    );
  }
}

