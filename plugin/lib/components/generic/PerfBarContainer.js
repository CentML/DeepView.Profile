'use babel';

import React from 'react';

class PerfBarContainer extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      marginTop: 0,
    };
    this._updateMarginTop = this._updateMarginTop.bind(this);
  }

  _updateMarginTop(marginTop) {
    this.setState({marginTop});
  }

  _classes() {
    const mainClass = 'innpv-perfbarcontainer-wrap';
    if (this.props.disabled) {
      return mainClass + ' innpv-no-events';
    }
    return mainClass;
  }

  render() {
    return (
      <div className={this._classes()}>
        <div className="innpv-perfbarcontainer">
          <div
            className="innpv-perfbarcontainer-inner"
            style={{marginTop: `-${this.state.marginTop}px`}}
          >
            {this.props.data.map(
              (dataElement, index) => this.props.perfBarGenerator(
                dataElement,
                index,
                this._updateMarginTop,
              ))}
          </div>
        </div>
        <LabelContainer labels={this.props.labels} />
      </div>
    );
  }
}

PerfBarContainer.defaultProps = {
  data: [],
  perfBarGenerator: (dataElement, index, updateMarginTop) => null,
  disabled: false,
  labels: [],
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
