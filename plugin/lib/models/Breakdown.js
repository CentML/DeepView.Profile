'use babel';

import {processFileReference} from '../utils';

export class Breakdown {
  constructor({
    operationTree,
    weightTree,
    peakUsageBytes,
    memoryCapacityBytes,
    iterationRunTimeMs,
  }) {
    this._operationTree = operationTree;
    this._weightTree = weightTree;
    this._peakUsageBytes = peakUsageBytes;
    this._memoryCapacityBytes = memoryCapacityBytes;
    this._iterationRunTimeMs = iterationRunTimeMs;
  }

  get operationTree() {
    return this._operationTree;
  }

  get weightTree() {
    return this._weightTree;
  }

  get peakUsageBytes() {
    return this._peakUsageBytes;
  }

  get memoryCapacityBytes() {
    return this._memoryCapacityBytes;
  }

  get iterationRunTimeMs() {
    return this._iterationRunTimeMs;
  }

  static fromBreakdownResponse(breakdownResponse) {
    return new Breakdown({
      operationTree: constructTree(
        breakdownResponse.getOperationTreeList(),
        OperationNode.fromProtobuf,
      ),
      weightTree: constructTree(
        breakdownResponse.getWeightTreeList(),
        WeightNode.fromProtobuf,
      ),
      peakUsageBytes:
        breakdownResponse.getPeakUsageBytes(),
      memoryCapacityBytes:
        breakdownResponse.getMemoryCapacityBytes(),
      iterationRunTimeMs:
        breakdownResponse.getIterationRunTimeMs(),
    });
  }
};

class BreakdownNode {
  constructor({id, name, contexts}) {
    this._id = id;
    this._name = name;
    this._contexts = contexts;
    this._parent = null;
    this._children = [];
  }

  get id() {
    return this._id;
  }

  get name() {
    return this._name;
  }

  get contexts() {
    return this._contexts;
  }

  get parent() {
    return this._parent;
  }

  get children() {
    return this._children;
  }
}

export class OperationNode extends BreakdownNode {
  constructor({id, name, contexts, forwardMs, backwardMs, sizeBytes}) {
    super({id, name, contexts});
    this._forwardMs = forwardMs;
    this._backwardMs = backwardMs;
    this._sizeBytes = sizeBytes;
  }

  get runTimeMs() {
    if (isNaN(this._backwardMs)) {
      return this._forwardMs;
    }
    return this._forwardMs + this._backwardMs;
  }

  get forwardMs() {
    return this._forwardMs;
  }

  get backwardMs() {
    return this._backwardMs;
  }

  get sizeBytes() {
    return this._sizeBytes;
  }

  static fromProtobuf(protobufBreakdownNode, id) {
    return new OperationNode({
      id,
      name: protobufBreakdownNode.getName(),
      contexts: protobufBreakdownNode.getContextsList().map(processFileReference),
      ...parseOperationData(protobufBreakdownNode),
    });
  }
};

export class WeightNode extends BreakdownNode {
  constructor({id, name, contexts, sizeBytes, gradSizeBytes}) {
    super({id, name, contexts});
    this._sizeBytes = sizeBytes;
    this._gradSizeBytes = gradSizeBytes;
  }

  get sizeBytes() {
    return this._sizeBytes;
  }

  get gradSizeBytes() {
    return this._gradSizeBytes;
  }

  static fromProtobuf(protobufBreakdownNode, id) {
    return new WeightNode({
      id,
      name: protobufBreakdownNode.getName(),
      contexts: protobufBreakdownNode.getContextsList().map(processFileReference),
      ...parseWeightData(protobufBreakdownNode),
    });
  }
};

function constructTree(protobufArray, constructNode) {
  if (protobufArray.length == 0) {
    throw new Error('Skyline received an empty breakdown tree! Please file a bug report.');
  }

  const helperRoot = {_children: []};
  const parentNodeStack = [helperRoot];
  const numChildrenStack = [1];

  // The nodes were placed into the array based on a preorder traversal of the tree
  protobufArray.forEach((protobufNode, index) => {
    const parent = parentNodeStack.pop();
    const numChildren = numChildrenStack.pop();

    // We use the serialization index as an identifier for the node
    const node = constructNode(protobufNode, index);
    node._parent = parent;
    parent._children.push(node);

    if (numChildren > 1) {
      parentNodeStack.push(parent);
      numChildrenStack.push(numChildren - 1);
    }

    if (protobufNode.getNumChildren() > 0 ) {
      parentNodeStack.push(node);
      numChildrenStack.push(protobufNode.getNumChildren());
    }
  });

  const root = helperRoot._children[0];
  root._parent = null;
  return root;
}

function parseOperationData(protobufNode) {
  const data = protobufNode.getOperation();
  return {
    forwardMs: data.getForwardMs(),
    backwardMs: data.getBackwardMs(),
    sizeBytes: data.getSizeBytes(),
  };
}

function parseWeightData(protobufNode) {
  const data = protobufNode.getWeight();
  return {
    sizeBytes: data.getSizeBytes(),
    gradSizeBytes: data.getGradSizeBytes(),
  };
}
