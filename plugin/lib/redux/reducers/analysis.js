'use babel';

import {
  ANALYSIS_REQ,
  ANALYSIS_REC_RUN,
  ANALYSIS_REC_MEM,
  ANALYSIS_REC_BRK,
  ANALYSIS_REC_THPT,
  ANALYSIS_ERROR,
} from '../actions/types';
import PerfVisState from '../../models/PerfVisState';
import MemoryBreakdown from '../../models/MemoryBreakdown';
import MemoryUsage from '../../models/MemoryUsage';
import Throughput from '../../models/Throughput';
import RunTimeBreakdown from '../../models/RunTimeBreakdown';
import {
  OperationNode,
  WeightNode,
} from '../../models/Breakdown';

export default function(state, action) {
  switch (action.type) {
    case ANALYSIS_REQ:
      return {
        ...state,
        perfVisState: PerfVisState.ANALYZING,
      };

    case ANALYSIS_REC_RUN: {
      const {runTimeResponse} = action.payload;
      return {
        ...state,
        runTimeBreakdown: RunTimeBreakdown.fromRunTimeResponse(runTimeResponse),
        errorMessage: '',
      };
    }

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

    case ANALYSIS_REC_BRK: {
      const {breakdownResponse} = action.payload;

      // Build the trees
      const operationTree = OperationNode.fromProtobufNodeList(
        breakdownResponse.getOperationTreeList(),
      );
      const weightTree = WeightNode.fromProtobufNodeList(
        breakdownResponse.getWeightTreeList(),
      );

      // Memory limits
      const peakUsageBytes = breakdownResponse.getPeakUsageBytes();
      const trackedBytes = operationTree.sizeBytes + weightTree.sizeBytes;
      const untrackedBytes = Math.max(0, peakUsageBytes - trackedBytes);

      // Run time limits
      const iterationRunTimeMs = breakdownResponse.getIterationRunTimeMs();
      const untrackedMs = Math.max(
        0.,
        iterationRunTimeMs - operationTree.runTimeMs,
      );

      return {
        ...state,
        breakdown: {
          operationTree,
          weightTree,
          currentView: null,
        },
        runTime: {
          untrackedMs,
          untrackedNode:
            untrackedMs > 0
              ? createUntrackedOperationNode({forwardMs: untrackedMs})
              : null,
        },
        memory: {
          untrackedBytes,
          untrackedNode:
            untrackedBytes > 0
              ? createUntrackedOperationNode({sizeBytes: untrackedBytes})
              : null,
        },
        peakUsageBytes,
        memoryCapacityBytes: breakdownResponse.getMemoryCapacityBytes(),
        iterationRunTimeMs,
      };
    }

    case ANALYSIS_REC_THPT: {
      const {throughputResponse} = action.payload;
      return {
        ...state,
        throughput:
          Throughput.fromThroughputResponse(throughputResponse),
        errorMessage: '',
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

function createUntrackedOperationNode(overrides) {
  return new OperationNode({
    id: -1,
    name: 'Untracked',
    forwardMs: 0.,
    backwardMs: 0.,
    sizeBytes: 0,
    contexts: [],
    ...overrides,
  });
}
