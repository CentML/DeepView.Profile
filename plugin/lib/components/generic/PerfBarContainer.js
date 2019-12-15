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
    const {children, marginTop, labels} = this.props;
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
        <LabelContainer labels={labels} />
      </div>
    );
  }
}

PerfBarContainer.defaultProps = {
  disabled: false,
  labels: [],
  marginTop: 0,
};

function LabelContainer(props) {
  if (props.labels.length === 0) {
    return null;
  }

  return (
    <div className="innpv-perfbarcontainer-labelcontainer">
      {props.labels.map(({label, percentage}) => (
        <div className="innpv-perfbarcontainer-labelwrap" style={{height: `${percentage}%`}}>
          {percentage >= 10 ? <div className="innpv-perfbarcontainer-label">{label}</div> : null}
        </div>
      ))}
    </div>
  );
}

export default PerfBarContainer;
