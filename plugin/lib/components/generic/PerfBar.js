'use babel';

import React from 'react';

import Elastic from './Elastic';
import PerfHintState from '../../models/PerfHintState';

class PerfBar extends React.Component {
  constructor(props) {
    super(props);
    this._tooltip = null;
    this._barRef = React.createRef();

    this.state = {
      // Keeps track of how this PerfBar is being manipulated.
      perfHintState: PerfHintState.NONE,

      // Keeps track of whether this PerfBar is "active". Right now, "active"
      // means the user's cursor is hovering over this PerfBar.
      isActive: false,
    };

    this._handleHoverEnter = this._handleHoverEnter.bind(this);
    this._handleHoverExit = this._handleHoverExit.bind(this);

    this._handleIncrease = this._handleIncrease.bind(this);
    this._handleDecrease = this._handleDecrease.bind(this);
    this._handleRestore = this._handleRestore.bind(this);
  }

  componentDidMount() {
    this._registerTooltip();
  }

  componentDidUpdate(prevProps) {
    this._reconcileTooltip(prevProps);
  }

  componentWillUnmount() {
    this._clearTooltip();
  }

  _reconcileTooltip(prevProps) {
    const {tooltipHTML} = this.props;
    const prevTooltipHTML = prevProps.tooltipHTML;
    if (tooltipHTML === prevTooltipHTML) {
      return;
    }
    this._clearTooltip();
    this._registerTooltip();
  }

  _registerTooltip() {
    const {tooltipHTML} = this.props;
    if (tooltipHTML == null) {
      return;
    }

    this._tooltip = atom.tooltips.add(
      this._barRef.current,
      {
        title: tooltipHTML,
        placement: 'right',
        html: true,
      },
    )
  }

  _clearTooltip() {
    if (this._tooltip == null) {
      return;
    }
    this._tooltip.dispose();
    this._tooltip = null;
  }

  _handleHoverEnter() {
    this.setState({isActive: true});
  }

  _handleHoverExit() {
    this.setState({isActive: false});
  }

  _handleIncrease() {
    if (this.state.perfHintState === PerfHintState.INCREASING) {
      return;
    }
    this.setState({perfHintState: PerfHintState.INCREASING});
  }

  _handleDecrease() {
    if (this.state.perfHintState === PerfHintState.DECREASING) {
      return;
    }
    this.setState({perfHintState: PerfHintState.DECREASING});
  }

  _handleRestore() {
    if (this.state.perfHintState === PerfHintState.NONE) {
      return;
    }
    this.setState({perfHintState: PerfHintState.NONE});
  }

  _className() {
    const {resizable, clickable} = this.props;
    const mainClass = 'innpv-perfbar-wrap';

    if (resizable) {
      return mainClass + ' innpv-perfbar-resizable';
    } else if (clickable) {
      return mainClass + ' innpv-perfbar-clickable';
    } else {
      return mainClass;
    }
  }

  render() {
    const {
      renderPerfHints,
      resizable,
      percentage,
      updateMarginTop,
      colorClass,
      onClick,
    } = this.props;
    const {isActive, perfHintState} = this.state;

    return (
      <Elastic
        className={this._className()}
        disabled={!resizable}
        heightPct={percentage}
        updateMarginTop={updateMarginTop}
        handleShrink={this._handleDecrease}
        handleGrow={this._handleIncrease}
        handleSnapBack={this._handleRestore}
      >
        <div
          ref={this._barRef}
          className={`innpv-perfbar ${colorClass}`}
          onMouseEnter={this._handleHoverEnter}
          onMouseLeave={this._handleHoverExit}
          onClick={onClick}
        />
        {renderPerfHints(isActive, perfHintState)}
      </Elastic>
    );
  }
}

PerfBar.defaultProps = {
  resizable: false,
  clickable: false,
  renderPerfHints: (isActive, perfHintState) => null,
  tooltipHTML: null,
  onClick: () => {},
};

export default PerfBar;
