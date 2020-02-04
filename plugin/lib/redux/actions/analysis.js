'use babel';

import {emptyFor, fromPayloadCreator} from './utils';
import {
  ANALYSIS_REQ,
  ANALYSIS_REC_RUN,
  ANALYSIS_REC_MEM,
  ANALYSIS_REC_BRK,
  ANALYSIS_REC_THPT,
  ANALYSIS_ERROR,
  ANALYSIS_EXPLORE_OP,
  ANALYSIS_EXPLORE_WEIGHT,
} from './types';

export default {
  request: emptyFor(ANALYSIS_REQ),
  receivedRunTimeAnalysis: fromPayloadCreator(
    ANALYSIS_REC_RUN,
    ({runTimeResponse}) => ({runTimeResponse}),
  ),
  receivedMemoryAnalysis: fromPayloadCreator(
    ANALYSIS_REC_MEM,
    ({memoryUsageResponse}) => ({memoryUsageResponse}),
  ),
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
};

