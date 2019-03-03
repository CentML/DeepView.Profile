'use babel';

import React from 'react';

import Subheader from './Subheader';

export default class Memory extends React.Component {
  render() {
    return (
      <div className="innpv-memory innpv-subpanel">
        <Subheader icon="database">Peak Memory Usage</Subheader>
      </div>
    );
  }
}

