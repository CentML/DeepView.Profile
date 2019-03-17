'use babel';

import React from 'react'

import SourceMarker from '../marker';
import INNPVStore from '../stores/innpv_store';

class UsageHighlight extends React.Component {
  constructor(props) {
    super(props);
    this._marker = new SourceMarker(INNPVStore.getEditor());
  }

  componentDidMount() {
    this._marker.register(this.props.location);
  }

  componentDidUpdate(prevProps) {
    this._marker.reconcileLocation(prevProps.location, this.props.location);
    if (!prevProps.show && this.props.show) {
      this._showDecoration();
    } else if (prevProps.show && !this.props.show) {
      this._hideDecoration();
    }
  }

  componentWillUnmount() {
    this._marker.remove();
  }

  _showDecoration() {
    this._marker.showDecoration({type: 'line', class: 'innpv-line-highlight'});
  }

  _hideDecoration() {
    this._marker.hideDecoration();
  }

  render() {
    return null;
  }
}

UsageHighlight.defaultProps = {
  show: false,
};

export default UsageHighlight;
