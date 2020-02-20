'use babel';

import path from 'path';
import React from 'react';
import {connect} from 'react-redux';

import ContextHighlightManager from './ContextHighlightManager';
import ErrorMessage from './ErrorMessage';
import Memory from './Memory';
import MemoryBreakdown from './MemoryBreakdown';
import PerfVisStatusBar from './PerfVisStatusBar';
import RunTimeBreakdown from './RunTimeBreakdown';
import Throughput from './Throughput';
import PerfVisState from '../models/PerfVisState';
import UsageHighlight from './UsageHighlight';

function PerfVisHeader() {
  return (
    <div className="innpv-header">
      <span className="icon icon-graph"></span>Skyline
    </div>
  );
}

class PerfVisMainView extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      showBatchSizeHighlight: false,
    };
    this._handleSliderHoverEnter = this._handleSliderHoverEnter.bind(this);
    this._handleSliderHoverExit = this._handleSliderHoverExit.bind(this);
    this._handleSliderClick = this._handleSliderClick.bind(this);
  }

  _handleSliderHoverEnter() {
    this.setState({
      showBatchSizeHighlight: true,
    });
  }

  _handleSliderHoverExit() {
    this.setState({
      showBatchSizeHighlight: false,
    });
  }

  _handleSliderClick() {
    const {batchSizeContext, projectRoot} = this.props;
    if (batchSizeContext == null || projectRoot == null) {
      return;
    }

    // Atom uses 0-based line numbers, but Skyline uses 1-based line numbers
    const absoluteFilePath = path.join(projectRoot, batchSizeContext.filePath);
    atom.workspace.open(
      absoluteFilePath,
      {initialLine: batchSizeContext.lineNumber - 1},
    );
  }

  _subrowClasses() {
    const {perfVisState, projectModified} = this.props;
    const mainClass = 'innpv-contents-subrows';
    if (projectModified || perfVisState === PerfVisState.ANALYZING) {
      return mainClass + ' innpv-no-events';
    }
    return mainClass;
  }

  _batchSizeUsageHighlight() {
    const {editorsByPath, batchSizeContext} = this.props;
    if (batchSizeContext == null ||
        !editorsByPath.has(batchSizeContext.filePath)) {
      return null;
    }

    const {showBatchSizeHighlight} = this.state;
    return editorsByPath.get(batchSizeContext.filePath).map((editor) => (
      <UsageHighlight
        key={editor.id}
        editor={editor}
        show={showBatchSizeHighlight}
        lineNumber={batchSizeContext.lineNumber}
      />
    ));
  }

  _renderBody() {
    const {
      perfVisState,
      projectRoot,
      errorMessage,
      projectModified,
    } = this.props;
    if (this.props.errorMessage !== '') {
      return (
        <ErrorMessage
          perfVisState={perfVisState}
          message={errorMessage}
          projectModified={projectModified}
        />
      );
    } else {
      return (
        <div className="innpv-contents-columns">
          <div className="innpv-perfbar-contents">
            <RunTimeBreakdown
              perfVisState={perfVisState}
              projectRoot={projectRoot}
              projectModified={projectModified}
            />
            <MemoryBreakdown
              perfVisState={perfVisState}
              projectRoot={projectRoot}
              projectModified={projectModified}
            />
          </div>
          <div className={this._subrowClasses()}>
            <Throughput
              handleSliderClick={this._handleSliderClick}
              handleSliderHoverEnter={this._handleSliderHoverEnter}
              handleSliderHoverExit={this._handleSliderHoverExit}
            />
            <Memory
              handleSliderClick={this._handleSliderClick}
              handleSliderHoverEnter={this._handleSliderHoverEnter}
              handleSliderHoverExit={this._handleSliderHoverExit}
            />
          </div>
        </div>
      );
    }
  }

  render() {
    const {perfVisState, projectModified} = this.props;
    return (
      <div className="innpv-main">
        <PerfVisHeader />
        <div className="innpv-contents">{this._renderBody()}</div>
        <PerfVisStatusBar
          perfVisState={perfVisState}
          projectModified={projectModified}
        />
        <ContextHighlightManager perfVisState={perfVisState} />
        {this._batchSizeUsageHighlight()}
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => ({
  editorsByPath: state.editorsByPath,
  batchSizeContext: state.batchSizeContext,
  projectModified: state.projectModified,
  ...ownProps,
});

export default connect(mapStateToProps)(PerfVisMainView);
