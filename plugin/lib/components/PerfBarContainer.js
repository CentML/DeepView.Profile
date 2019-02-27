'use babel';

import React from 'react';

import PerfBar from './PerfBar';

const COLOR_CLASSES = [
  'ui-site-1',
  'ui-site-2',
  'ui-site-3',
  'ui-site-4',
  'ui-site-5',
];

class PerfBarContainer extends React.Component {
  render() {
    const totalTimeMicro =
      this.props.operationInfos.reduce((acc, info) => acc + info.getRuntimeUs(), 0);

    return (
      <div className="innpv-perfbarcontainer">
        {this.props.operationInfos.map(
          (operationInfo, index) =>
            <PerfBar
              key={operationInfo.getBoundName()}
              operationInfo={operationInfo}
              percentage={operationInfo.getRuntimeUs() / totalTimeMicro * 100}
              colorClass={COLOR_CLASSES[index % COLOR_CLASSES.length]}
            />
        )}
      </div>
    );
  }
}

PerfBarContainer.defaultProps = {
  operationInfos: [],
}

export default PerfBarContainer;
