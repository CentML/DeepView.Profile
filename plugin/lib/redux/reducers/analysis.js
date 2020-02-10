'use babel';

import {
  ANALYSIS_REQ,
  ANALYSIS_REC_BRK,
  ANALYSIS_REC_THPT,
  ANALYSIS_ERROR,
  ANALYSIS_EXPLORE_OP,
  ANALYSIS_EXPLORE_WEIGHT,
  ANALYSIS_EXPLORE_CLEAR,
  ANALYSIS_EXPLORE_PREV,
  ANALYSIS_SET_ACTIVE,
} from '../actions/types';
import PerfVisState from '../../models/PerfVisState';
import Throughput from '../../models/Throughput';
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
      const iterationRunTimeMs = Math.max(
        breakdownResponse.getIterationRunTimeMs(),
        operationTree.runTimeMs,
      );
      const untrackedMs = iterationRunTimeMs - operationTree.runTimeMs;

      return {
        ...state,
        breakdown: {
          operationTree,
          weightTree,
          currentView: null,
          currentlyActive: null,
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

    case ANALYSIS_EXPLORE_OP:
    case ANALYSIS_EXPLORE_WEIGHT: {
      const {newView} = action.payload;
      const perfVisState = action.type === ANALYSIS_EXPLORE_OP
        ? PerfVisState.EXPLORING_OPERATIONS
        : PerfVisState.EXPLORING_WEIGHTS;
      return {
        ...state,
        perfVisState,
        breakdown: {
          ...state.breakdown,
          currentView: newView,
          currentlyActive: null,
        },
      };
    }

    case ANALYSIS_EXPLORE_CLEAR: {
      return {
        ...state,
        perfVisState: PerfVisState.READY,
        breakdown: {
          ...state.breakdown,
          currentView: null,
        },
      };
    }

    case ANALYSIS_EXPLORE_PREV: {
      const currentView = state.breakdown.currentView;
      if (currentView.parent !== state.breakdown.operationTree &&
          currentView.parent !== state.breakdown.weightTree) {
        return {
          ...state,
          breakdown: {
            ...state.breakdown,
            currentView: currentView.parent,
          },
        };
      }

      return {
        ...state,
        perfVisState: PerfVisState.READY,
        breakdown: {
          ...state.breakdown,
          currentView: null,
        },
      };
    }

    case ANALYSIS_SET_ACTIVE: {
      const {currentlyActive} = action.payload;
      return {
        ...state,
        breakdown: {
          ...state.breakdown,
          currentlyActive,
        },
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
