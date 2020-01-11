'use babel';

import {
  ANALYSIS_REQ,
  ANALYSIS_REC_MEM,
  ANALYSIS_REC_THPT,
  ANALYSIS_ERROR,
} from '../actions/types';
import PerfVisState from '../../models/PerfVisState';
import MemoryBreakdown from '../../models/MemoryBreakdown';
import MemoryUsage from '../../models/MemoryUsage';
import Throughput from '../../models/Throughput';

export default function(state, action) {
  switch (action.type) {
    case ANALYSIS_REQ:
      return {
        ...state,
        perfVisState: PerfVisState.ANALYZING,
      };

    case ANALYSIS_REC_MEM: {
      const {memoryUsageResponse} = action.payload;
      return {
        ...state,
        memoryBreakdown:
          MemoryBreakdown.fromMemoryUsageResponse(memoryUsageResponse),
        memoryUsage:
          MemoryUsage.fromMemoryUsageResponse(memoryUsageResponse),
        errorMessage: '',
      };
    }

    case ANALYSIS_REC_THPT: {
      const {throughputResponse} = action.payload;
      return {
        ...state,
        throughput:
          Throughput.fromThroughputResponse(throughputResponse),
        perfVisState:
          state.perfVisState !== PerfVisState.MODIFIED
            ? PerfVisState.READY
            : PerfVisState.MODIFIED,
      };
    }

    case ANALYSIS_ERROR:
      return {
        ...state,
        errorMessage: action.payload.errorMessage,
        perfVisState: PerfVisState.ERROR,
      };

    default:
      return state;
  }
};
