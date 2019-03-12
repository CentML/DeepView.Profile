'use babel';

import React from 'react';

import Subheader from './Subheader';
import BarSlider from './BarSlider';
import NumericDisplay from './NumericDisplay';
import BatchSizeStore from '../stores/batchsize_store';
import INNPVStore from '../stores/innpv_store';
import PerfVisState from '../models/PerfVisState';

export default class Throughput extends React.Component {
  constructor(props) {
    super(props);
    this._handleResize = this._handleResize.bind(this);
  }

  _handleResize(deltaPct, basePct) {
    // TODO: Use the new batch size to modify the code
    const newBatch = BatchSizeStore.updateThroughput(deltaPct, basePct);
    INNPVStore.setPerfVisState(PerfVisState.SHOWING_PREDICTIONS);
  }

  render() {
    const {model, handleSliderHoverEnter, handleSliderHoverExit} = this.props;
    const notReady = model == null;

    return (
      <div className="innpv-throughput innpv-subpanel">
        <Subheader icon="flame">Training Throughput</Subheader>
        <div className="innpv-subpanel-content">
          <BarSlider
            percentage={notReady ? 0 : model.displayPct}
            limitPercentage={notReady ? 100 : model.limitPct}
            handleResize={this._handleResize}
            onMouseEnter={handleSliderHoverEnter}
            onMouseLeave={handleSliderHoverExit}
          />
          <div className="innpv-subpanel-sidecontent">
            <NumericDisplay
              top="Throughput"
              number={notReady ? '---' : model.throughput}
              bottom="samples/second"
            />
            <div className="innpv-separator" />
            <NumericDisplay
              top="Theoretical Maximum"
              number={notReady ? '---' : model.maxThroughput}
              bottom="samples/second"
            />
          </div>
        </div>
      </div>
    );
  }
}
