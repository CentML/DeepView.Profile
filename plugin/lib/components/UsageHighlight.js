'use babel';

import React from 'react';

import InlineHighlight from './generic/InlineHighlight';

export default function UsageHighlight(props) {
  return (
    <InlineHighlight
      decorations={[{type: 'line', class: 'innpv-line-highlight'}]}
      {...props}
    />
  );
}
