'use babel';

import {emptyFor, fromPayloadCreator} from './utils';
import {
  ANALYSIS_REQ,
  ANALYSIS_REC_MEM,
  ANALYSIS_REC_THPT,
  ANALYSIS_ERROR,
} from './types';

export default {
  request: emptyFor(ANALYSIS_REQ),
  receivedMemoryAnalysis: fromPayloadCreator(
    ANALYSIS_REC_MEM,
    ({memoryUsageResponse}) => ({memoryUsageResponse}),
  ),
  receivedThroughputAnalysis: fromPayloadCreator(
    ANALYSIS_REC_THPT,
    ({throughputResponse}) => ({throughputResponse}),
  ),
  error: fromPayloadCreator(ANALYSIS_ERROR, ({errorMessage}) => ({errorMessage})),
};

