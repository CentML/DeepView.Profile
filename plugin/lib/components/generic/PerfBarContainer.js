'use babel';

import React from 'react';

class PerfBarContainer extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      expanded: null,
    };
    this._onLabelClick = this._onLabelClick.bind(this);
  }

  _onLabelClick(label) {
    const {expanded} = this.state;
    const {labels} = this.props;

    if (expanded == null &&
        labels.some((labelInfo) => labelInfo.clickable && labelInfo.label === label)) {
      this.setState({expanded: label});
    } else if (expanded === label) {
      this.setState({expanded: null});
    }
  }

  _classes() {
    const mainClass = 'innpv-perfbarcontainer-wrap';
    if (this.props.disabled) {
      return mainClass + ' innpv-no-events';
    }
    return mainClass;
  }

  render() {
    const {renderPerfBars, marginTop, labels} = this.props;
    const {expanded} = this.state;
    return (
      <div className={this._classes()}>
        <div className="innpv-perfbarcontainer">
          <div
            className="innpv-perfbarcontainer-inner"
            style={{marginTop: `-${marginTop}px`}}
          >
            {renderPerfBars(expanded)}
          </div>
        </div>
        <LabelContainer
          labels={labels}
          expanded={expanded}
          onLabelClick={this._onLabelClick}
        />
      </div>
    );
  }
}

PerfBarContainer.defaultProps = {
  disabled: false,
  labels: [],
  marginTop: 0,
  renderPerfBars: (expanded) => null,
};

function LabelContainer(props) {
  if (props.labels.length === 0) {
    return null;
  }

  return (
    <div className="innpv-perfbarcontainer-labelcontainer">
      {props.labels.filter(({percentage}) => percentage > 0).map(({label, percentage, clickable}) => {
        let displayPct = percentage;
        if (props.expanded != null) {
          if (label === props.expanded) {
            displayPct = 100;
          } else {
            displayPct = 0.001;
          }
        }
        return (
          <div
            className={
              `innpv-perfbarcontainer-labelwrap ${clickable ? 'innpv-perfbarcontainer-clickable' : ''}`
            }
            key={label}
            style={{height: `${displayPct}%`}}
            onClick={() => props.onLabelClick(label)}
          >
            <div className="innpv-perfbarcontainer-label">{displayPct >= 5 ? label : null}</div>
          </div>
        );
      })}
    </div>
  );
}

export default PerfBarContainer;
