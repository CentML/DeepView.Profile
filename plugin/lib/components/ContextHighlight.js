'use babel';

import React from 'react';

import InlineHighlight from './generic/InlineHighlight';

const LINE_NUMBER_DECORATION = {
  type: 'line-number',
  class: 'innpv-contexthighlight-linenum',
};

export default class ContextHighlight extends React.Component {
  constructor(props) {
    super(props);
    this._element = document.createElement('div');
  }

  render() {
    const {editor, lineNumber} = this.props;
    return (
      <InlineHighlight
        editor={editor}
        decorations={[LINE_NUMBER_DECORATION]}
        lineNumber={lineNumber}
        show={true}
      />
    );
  }
};
