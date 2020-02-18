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
  ANALYSIS_DRAG_THPT,
  ANALYSIS_DRAG_MEM,
  ANALYSIS_PRED_CLEAR,
} from '../actions/types';
import PerfVisState from '../../models/PerfVisState';
import Throughput from '../../models/Throughput';
import LinearModel from '../../models/LinearModel';
import {
  OperationNode,
  WeightNode,
} from '../../models/Breakdown';
import {processFileReference} from '../../utils';

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
        batchSize: breakdownResponse.getBatchSize(),
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
        predictionModels: {
          ...state.predictionModels,
          currentBatchSize: null,
        },
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
      const runTimeMsModel = LinearModel.fromProtobuf(
        throughputResponse.getRunTimeMs(),
      );
      const peakUsageBytesModel = LinearModel.fromProtobuf(
        throughputResponse.getPeakUsageBytes(),
      );
      return {
        ...state,
        throughput:
          Throughput.fromThroughputResponse(throughputResponse),
        predictionModels: {
          runTimeMs: runTimeMsModel,
          peakUsageBytes: peakUsageBytesModel,
          currentBatchSize: null,
          maxBatchSize: Math.max(getBatchSizeFromUsage(
            peakUsageBytesModel,
            state.memoryCapacityBytes,
          ), 1),
          batchSizeManipulatable:
            throughputResponse.getCanManipulateBatchSize(),
        },
        batchSizeContext:
          processFileReference(throughputResponse.getBatchSizeContext()),
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

    case ANALYSIS_DRAG_THPT: {
      if (!state.throughput.hasMaxThroughputPrediction) {
        return state;
      }

      // Map the drag delta to a throughput value
      // NOTE: We clamp the values (upper bound for throughput, lower bound for
      //       batch size)
      const {basePct, deltaPct} = action.payload;
      const updatedPct = basePct + deltaPct;
      const updatedThroughputSeconds = Math.max(Math.min(
        updatedPct / 100 * state.throughput.predictedMaxSamplesPerSecond,
        state.throughput.predictedMaxSamplesPerSecond,
      ), 0);
      const throughputBatchSize = getBatchSizeFromThroughput(
        state.predictionModels.runTimeMs,
        updatedThroughputSeconds,
      );

      let predictedBatchSize;
      if (throughputBatchSize < 0) {
        // NOTE: The throughput batch size may be so large that it overflows
        predictedBatchSize = state.predictionModels.maxBatchSize;
      } else {
        predictedBatchSize = Math.max(Math.min(
          throughputBatchSize, state.predictionModels.maxBatchSize), 1);
      }

      return {
        ...state,
        perfVisState: PerfVisState.SHOWING_PREDICTIONS,
        breakdown: {
          ...state.breakdown,
          currentView: null,
        },
        predictionModels: {
          ...state.predictionModels,
          currentBatchSize: predictedBatchSize,
        },
      };
    }

    case ANALYSIS_DRAG_MEM: {
      // Map the drag delta to a usage value
      // NOTE: We clamp the values (upper bound for usage, lower bound for
      //       batch size)
      const {deltaPct, basePct} = action.payload;
      const updatedPct = basePct + deltaPct;
      const updatedUsageBytes = Math.min(
        updatedPct / 100 * state.memoryCapacityBytes,
        state.memoryCapacityBytes,
      );

      const predictedBatchSize = Math.min(Math.max(
        getBatchSizeFromUsage(
          state.predictionModels.peakUsageBytes,
          updatedUsageBytes,
        ),
        1,
      ), state.predictionModels.maxBatchSize);

      return {
        ...state,
        perfVisState: PerfVisState.SHOWING_PREDICTIONS,
        breakdown: {
          ...state.breakdown,
          currentView: null,
        },
        predictionModels: {
          ...state.predictionModels,
          currentBatchSize: predictedBatchSize,
        },
      };
    }

    case ANALYSIS_PRED_CLEAR: {
      return {
        ...state,
        perfVisState: PerfVisState.READY,
        predictionModels: {
          ...state.predictionModels,
          currentBatchSize: null,
        },
      };
    }

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

function getBatchSizeFromUsage(peakUsageBytesModel, usageBytes) {
  return (usageBytes - peakUsageBytesModel.bias) / peakUsageBytesModel.slope;
}

function getBatchSizeFromThroughput(runTimeMsModel, samplesPerSecond) {
  const throughputMs = samplesPerSecond / 1000;
  return (throughputMs * runTimeMsModel.bias) /
    (1 - throughputMs * runTimeMsModel.slope);
}
