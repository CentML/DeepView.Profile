'use babel';

import React from 'react';

export default class Subheader extends React.Component {
  render() {
    const {icon, children} = this.props;
    return (
      <div className="innpv-subheader">
        <span className={`icon icon-${icon}`} />
        {this.props.children}
      </div>
    );
  }
}
