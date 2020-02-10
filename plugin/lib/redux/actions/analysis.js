'use babel';

import {emptyFor, fromPayloadCreator} from './utils';
import {
  ANALYSIS_REQ,
  ANALYSIS_REC_BRK,
  ANALYSIS_REC_THPT,
  ANALYSIS_ERROR,
  ANALYSIS_EXPLORE_OP,
  ANALYSIS_EXPLORE_WEIGHT,
  ANALYSIS_EXPLORE_PREV,
  ANALYSIS_EXPLORE_CLEAR,
  ANALYSIS_SET_ACTIVE,
} from './types';

export default {
  request: emptyFor(ANALYSIS_REQ),
  receivedBreakdown: fromPayloadCreator(
    ANALYSIS_REC_BRK,
    ({breakdownResponse}) => ({breakdownResponse}),
  ),
  receivedThroughputAnalysis: fromPayloadCreator(
    ANALYSIS_REC_THPT,
    ({throughputResponse}) => ({throughputResponse}),
  ),
  error: fromPayloadCreator(
    ANALYSIS_ERROR,
    ({errorMessage}) => ({errorMessage}),
  ),
  exploreOperation: fromPayloadCreator(
    ANALYSIS_EXPLORE_OP,
    ({newView}) => ({newView}),
  ),
  exploreWeight: fromPayloadCreator(
    ANALYSIS_EXPLORE_WEIGHT,
    ({newView}) => ({newView}),
  ),
  explorePrevious: emptyFor(ANALYSIS_EXPLORE_PREV),
  clearExplored: emptyFor(ANALYSIS_EXPLORE_CLEAR),
  setActive: fromPayloadCreator(
    ANALYSIS_SET_ACTIVE,
    ({currentlyActive}) => ({currentlyActive}),
  ),
};

