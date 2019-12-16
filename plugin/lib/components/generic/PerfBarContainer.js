'use babel';

import React from 'react';

class PerfBarContainer extends React.Component {
  _classes() {
    const mainClass = 'innpv-perfbarcontainer-wrap';
    if (this.props.disabled) {
      return mainClass + ' innpv-no-events';
    }
    return mainClass;
  }

  render() {
    const {children, marginTop, labels, onLabelClick} = this.props;
    return (
      <div className={this._classes()}>
        <div className="innpv-perfbarcontainer">
          <div
            className="innpv-perfbarcontainer-inner"
            style={{marginTop: `-${marginTop}px`}}
          >
            {children}
          </div>
        </div>
        <LabelContainer labels={labels} onLabelClick={onLabelClick} />
      </div>
    );
  }
}

PerfBarContainer.defaultProps = {
  disabled: false,
  labels: [],
  marginTop: 0,
  onLabelClick: () => {},
};

function LabelContainer(props) {
  if (props.labels.length === 0) {
    return null;
  }

  return (
    <div className="innpv-perfbarcontainer-labelcontainer">
      {props.labels.filter(({percentage}) => percentage > 0).map(({label, percentage}) => (
        <div className="innpv-perfbarcontainer-labelwrap"
          key={label}
          style={{height: `${percentage}%`}}
          onClick={() => props.onLabelClick(label)}
        >
          <div className="innpv-perfbarcontainer-label">{percentage >= 5 ? label : null}</div>
        </div>
      ))}
    </div>
  );
}

export default PerfBarContainer;
