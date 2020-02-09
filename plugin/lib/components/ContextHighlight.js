'use babel';

import React from 'react';
import ReactDOM from 'react-dom';

import InlineHighlight from './generic/InlineHighlight';
import {toPercentage} from '../utils';

export default class ContextHighlight extends React.Component {
  constructor(props) {
    super(props);
    this._gutterElement = document.createElement('div');
    this._overlayElement = document.createElement('div');
    this._decorationOptions = [
      {
        type: 'overlay',
        item: this._overlayElement,
        class: 'innpv-contexthighlight-overlay',
      },
      {
        type: 'gutter',
        item: this._gutterElement,
      },
    ];
    this.state = {
      showDisplay: false,
    };
    this._onMouseEnter = this._onMouseEnter.bind(this);
    this._onMouseLeave = this._onMouseLeave.bind(this);
  }

  _onMouseEnter() {
    this.setState({showDisplay: true});
  }

  _onMouseLeave() {
    this.setState({showDisplay: false});
  }

  render() {
    const {Fragment} = React;
    const {
      editor,
      lineNumber,
      isScoped,
      scopedContextInfo,
      ...rest,
    } = this.props;
    // We make this gutter marker "inactive" (greyed out) if the user is
    // currently exploring a specific "scope" (part of the breakdown tree) AND
    // we do not have specific scoped information for this particular line of
    // code.
    const inactiveGutterMarker = isScoped && scopedContextInfo == null;

    return (
      <Fragment>
        <InlineHighlight
          editor={editor}
          decorations={this._decorationOptions}
          lineNumber={lineNumber}
          column={999 /* HACK: Atom places this at the "end" of the line. */}
          show={true}
        />
        {this.state.showDisplay
          ? ReactDOM.createPortal(
              <ContextDisplay
                isScoped={isScoped}
                scopedContextInfo={scopedContextInfo}
                {...rest}
              />,
              this._overlayElement,
            )
          : null}
        {ReactDOM.createPortal(
          <GutterMarker
            onMouseEnter={this._onMouseEnter}
            onMouseLeave={this._onMouseLeave}
            inactive={inactiveGutterMarker}
          />,
          this._gutterElement,
        )}
      </Fragment>
    );
  }
};

class ContextDisplay extends React.Component {
  _renderDisplayInfo() {
    const {isScoped, scopedContextInfo, contextInfo} = this.props;
    return (
      <div className="innpv-contextmarker-displayinfo">
        <span><strong>Overall Invocations: </strong>{contextInfo.invocations}</span>
        {
          isScoped ? (
            <span>
              {" "}
              | <strong>Within Scope: </strong>
              {scopedContextInfo != null ? scopedContextInfo.invocations : 0}
            </span>
          ) : null
        }
      </div>
    );
  }

  render() {
    const {
      contextInfo,
      scopedContextInfo,
      iterationRunTimeMs,
      peakUsageBytes,
      isScoped,
    } = this.props;
    return (
      <div className="innpv-contextmarker-displaywrap">
        <div className="innpv-contextmarker-display">
          <div className="innpv-contextmarker-displaypointer" />
          <div className="innpv-contextmarker-displaycontent">
            {scopedContextInfo != null
              ? <ContextPerfView
                  title="Within Scope"
                  contextInfo={scopedContextInfo}
                  iterationRunTimeMs={iterationRunTimeMs}
                  peakUsageBytes={peakUsageBytes}
                />
              : null
            }
            <ContextPerfView
              title="Overall"
              contextInfo={contextInfo}
              iterationRunTimeMs={iterationRunTimeMs}
              peakUsageBytes={peakUsageBytes}
              faded={isScoped}
            />
            {this._renderDisplayInfo()}
          </div>
        </div>
      </div>
    );
  }
}

function ContextBar(props) {
  const {label, percentage} = props;
  return (
    <div className="innpv-contextmarker-barwrap">
      <div className="innpv-contextmarker-barlabel">{label}</div>
      <div className="innpv-contextmarker-bararea">
        <div
          className="innpv-contextmarker-bar"
          style={{width: `${percentage}%`}}
        />
        <div className="innpv-contextmarker-barnum">
          {percentage.toFixed(1) + '%'}
        </div>
      </div>
    </div>
  );
}

function ContextPerfView(props) {
  const {iterationRunTimeMs, peakUsageBytes, title} = props;
  const {runTimeMs, sizeBytes} = props.contextInfo;
  const runTimePct = toPercentage(runTimeMs, iterationRunTimeMs);
  const memoryPct = toPercentage(sizeBytes, peakUsageBytes);
  let className = 'innpv-contextmarker-perfview';
  if (props.faded) {
    className += ' innpv-contextmarker-faded';
  }

  return (
    <div className={className}>
      <div className="innpv-contextmarker-perfview-title">{title}</div>
      <ContextBar label="Run Time" percentage={runTimePct} />
      {memoryPct > 0.
        ? <ContextBar label="Memory" percentage={memoryPct} />
        : null}
    </div>
  );
}

function GutterMarker(props) {
  const {onMouseEnter, onMouseLeave, inactive} = props;
  let className = 'innpv-contexthighlight-guttermarker';
  if (inactive) {
    className += ' innpv-contexthighlight-guttermarker-inactive';
  }
  return (
    <div
      className={className}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div/>
    </div>
  );
}
