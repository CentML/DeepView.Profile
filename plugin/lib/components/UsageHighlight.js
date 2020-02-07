'use babel';

import React from 'react';

import InlineHighlight from './generic/InlineHighlight';

const DECORATIONS = [
  {type: 'line', class: 'innpv-line-highlight'},
];

export default function UsageHighlight(props) {
  return (
    <InlineHighlight
      decorations={DECORATIONS}
      {...props}
    />
  );
}
